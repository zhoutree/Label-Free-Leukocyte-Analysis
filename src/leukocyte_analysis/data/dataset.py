"""按划分清单读取打包图块，供 PyTorch 模型训练和评估。"""

import csv
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .bbbc045_config import CLASS_NAMES, model_label


class LeukocyteDataset(Dataset):
    """从无损图块数组中读取指定数据组的 55×55 明场图像。"""

    def __init__(
        self,
        split_csv: Path,
        tiles_npy: Path,
        stats_json: Path,
        split: str,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"未知数据组：{split}")

        self.class_names = CLASS_NAMES
        class_to_index = {name: index for index, name in enumerate(self.class_names)}
        self.tiles_npy = tiles_npy.resolve()
        self.transform = transform
        self._tiles = None  # 实际读取时才打开；各 DataLoader 进程分别映射。
        self.samples = []

        # CSV 行号与 .npy 图块索引一一对应，只保留当前数据组的行。
        with split_csv.open("r", newline="", encoding="utf-8") as csv_file:
            for packed_index, record in enumerate(csv.DictReader(csv_file)):
                if record["split"] == split:
                    # 原始清单仍保存 B/T；只在读取时合并为淋巴细胞。
                    label = class_to_index[model_label(record["label"])]
                    self.samples.append(
                        (packed_index, label, record["donor"])
                    )

        with stats_json.open("r", encoding="utf-8") as json_file:
            stats = json.load(json_file)

        self.mean = float(stats["mean"])
        self.std = float(stats["std"])
        if self.std <= 0 or not self.samples:
            raise ValueError("标准差无效或当前数据组没有样本")

    def __len__(self) -> int:
        return len(self.samples)

    def __getstate__(self) -> dict:
        """传给 Windows 的 DataLoader 子进程时，不复制已打开的图块映射。"""
        state = self.__dict__.copy()
        state["_tiles"] = None
        return state

    def _get_tiles(self) -> np.ndarray:
        if self._tiles is None:
            self._tiles = np.load(
                self.tiles_npy,
                mmap_mode="r",
                allow_pickle=False,
            )
        return self._tiles

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        packed_index, label, _donor = self.samples[index]
        tile = self._get_tiles()[packed_index]

        # 归一化得到 1×55×55 的 float32 张量，不修改原始 .npy 文件。
        normalized = ((tile - self.mean) / self.std).astype(
            np.float32,
            copy=True,
        )
        image = torch.from_numpy(normalized[None, :, :])
        # 增强只由训练加载器传入；验证和测试图像保持原样。
        if self.transform is not None:
            image = self.transform(image)
        return image, label
