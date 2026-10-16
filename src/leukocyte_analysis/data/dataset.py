"""按划分清单读取打包图块，供 PyTorch 模型训练和评估。"""

import csv
import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .bbbc045_config import CLASS_NAMES, model_label
from .integrity import validate_bundle


class LeukocyteDataset(Dataset):
    """从无损图块数组中读取指定数据组的 55×55 明场图像。"""

    def __init__(
        self,
        split_csv: Path,
        tiles_npy: Path,
        stats_json: Path,
        split: str,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        include_index: bool = False,
        raw_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """按 CSV 行号建立图块索引，并读取训练集统计量。

        ``raw_transform`` 在标准化前作用于原始灰度，``transform`` 在标准化后
        作用于模型输入；``include_index`` 供验证错分追溯原始图块。
        """
        if split not in {"train", "val", "test"}:
            raise ValueError(f"未知数据组：{split}")

        self.class_names = CLASS_NAMES
        class_to_index = {name: index for index, name in enumerate(self.class_names)}
        self.tiles_npy = tiles_npy.resolve()
        self.transform = transform
        self.raw_transform = raw_transform
        self.include_index = include_index
        self._tiles = None  # 实际读取时才打开；各 DataLoader 进程分别映射。
        self.samples = []
        self.source_by_index = {}
        record_count = 0

        # CSV 行号与 .npy 图块索引一一对应，只保留当前数据组的行。
        with split_csv.open("r", newline="", encoding="utf-8") as csv_file:
            for packed_index, record in enumerate(csv.DictReader(csv_file)):
                record_count += 1
                if record["split"] == split:
                    # 原始清单仍保存 B/T；只在读取时合并为淋巴细胞。
                    label = class_to_index[model_label(record["label"])]
                    self.samples.append(
                        (packed_index, label, record["donor"])
                    )
                    # 错分记录通过同一索引追溯原始 TIFF、网格位置和原始标签。
                    self.source_by_index[packed_index] = {
                        "file": record.get("file", ""),
                        "donor": record["donor"],
                        "source_label": record["label"],
                        "row": record.get("row", ""),
                        "col": record.get("col", ""),
                    }

        with stats_json.open("r", encoding="utf-8") as json_file:
            stats = json.load(json_file)

        self.mean = float(stats["mean"])
        self.std = float(stats["std"])
        if not math.isfinite(self.mean) or not math.isfinite(self.std) or self.std <= 0:
            raise ValueError("归一化均值或标准差无效")
        if not self.samples:
            raise ValueError("当前数据组没有样本")
        validate_bundle(split_csv, self.tiles_npy, stats, record_count)

    def __len__(self) -> int:
        """返回当前数据划分中的图块数。"""
        return len(self.samples)

    def __getstate__(self) -> dict:
        """传给 Windows 的 DataLoader 子进程时，不复制已打开的图块映射。"""
        state = self.__dict__.copy()
        state["_tiles"] = None
        return state

    def _get_tiles(self) -> np.ndarray:
        """在当前进程首次取样时打开只读图块映射。"""
        if self._tiles is None:
            self._tiles = np.load(
                self.tiles_npy,
                mmap_mode="r",
                allow_pickle=False,
            )
        return self._tiles

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int] | tuple[torch.Tensor, int, int]:
        """读取单通道图块和四分类标签，必要时附带原清单索引。"""
        packed_index, label, _donor = self.samples[index]
        tile = self._get_tiles()[packed_index]

        # 亮度/对比度先作用于原始灰度值，再按训练集统计量归一化。
        # 未启用时沿用原路径，使仅翻转基线的输入不变。
        if self.raw_transform is None:
            normalized = ((tile - self.mean) / self.std).astype(np.float32, copy=True)
            image = torch.from_numpy(normalized[None, :, :])
        else:
            image = torch.from_numpy(np.array(tile[None, :, :], dtype=np.float32, copy=True))
            image = self.raw_transform(image)
            image = (image - self.mean) / self.std
        # 几何增强只由训练加载器传入；验证和测试图像保持原样。
        if self.transform is not None:
            image = self.transform(image)
        if self.include_index:
            return image, label, packed_index
        return image, label
