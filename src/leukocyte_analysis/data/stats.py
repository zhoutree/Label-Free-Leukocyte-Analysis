"""从训练集原始像素计算归一化参数，不使用验证集或测试集。"""

import csv
import json
from pathlib import Path

import numpy as np


def write_train_stats(
    split_csv: Path, tiles_npy: Path, output_json: Path
) -> dict:
    """仅用 ``split=train`` 的原始像素分批计算全局统计量。

    返回并写出图块数、像素数、均值、总体标准差和极值；验证／测试
    图块不参与统计，供三组数据复用同一尺度。
    """
    with split_csv.open("r", newline="", encoding="utf-8") as csv_file:
        records = list(csv.DictReader(csv_file))

    tiles = np.load(tiles_npy, mmap_mode="r", allow_pickle=False)
    if len(records) != len(tiles):
        raise ValueError("CSV 行数与打包图块数不一致")

    train_indices = [
        index for index, record in enumerate(records)
        if record["split"] == "train"
    ]
    if not train_indices:
        raise ValueError("没有找到训练集图块")

    # 分批处理，避免一次把约 6.9 万张图全部载入内存。
    pixel_count = 0
    mean = 0.0
    m2 = 0.0
    minimum = float("inf")
    maximum = float("-inf")

    for start in range(0, len(train_indices), 256):
        indices = train_indices[start : start + 256]
        block = np.asarray(tiles[indices], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError("训练集中存在非有限像素值")

        block_count = block.size
        block_mean = float(block.mean())
        block_variance = float(block.var())

        # 合并当前批次与此前批次的均值、方差。
        new_count = pixel_count + block_count
        delta = block_mean - mean
        m2 += (
            block_variance * block_count
            + delta * delta * pixel_count * block_count / new_count
        )
        mean += delta * block_count / new_count
        pixel_count = new_count

        minimum = min(minimum, float(block.min()))
        maximum = max(maximum, float(block.max()))

    stats = {
        "train_tiles": len(train_indices),
        "pixel_count": pixel_count,
        "mean": mean,
        "std": (m2 / pixel_count) ** 0.5,
        "min": minimum,
        "max": maximum,
    }
    if stats["std"] <= 0:
        raise ValueError("训练集像素标准差必须大于 0")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("x", encoding="utf-8") as json_file:
        json.dump(stats, json_file, ensure_ascii=False, indent=2)

    return stats
