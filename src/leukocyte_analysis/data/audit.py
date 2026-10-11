"""审计打包数据的结构和基础图像质量，不修改或删除样本。"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .bbbc045_config import (
    SOURCE_CLASS_NAMES,
    TEST_DONORS,
    TILE_SIZE,
    TRAIN_DONORS,
    VAL_DONORS,
)
from .quality import focus_proxy


def _read_records(split_csv: Path) -> list[dict[str, str]]:
    """读取清单并检查必需列。"""
    with split_csv.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"file", "donor", "label", "row", "col", "split"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("划分清单缺少必需列")
        records = list(reader)

    if not records:
        raise ValueError("划分清单为空")
    return records


def _validate_records(records: list[dict[str, str]]) -> Counter:
    """检查实际受试者划分、类别和图块位置，并统计各组数量。"""
    donor_groups = {
        "train": TRAIN_DONORS,
        "val": VAL_DONORS,
        "test": TEST_DONORS,
    }
    counts = Counter()
    seen_positions = set()

    for index, record in enumerate(records):
        split = record["split"]
        label = record["label"]
        donor = record["donor"]

        if split not in donor_groups or donor not in donor_groups[split]:
            raise ValueError(f"第 {index} 行的受试者划分不正确：{donor}, {split}")
        if label not in SOURCE_CLASS_NAMES:
            raise ValueError(f"第 {index} 行类别未知：{label}")

        row, col = int(record["row"]), int(record["col"])
        if row < 0 or col < 0:
            raise ValueError(f"第 {index} 行的图块坐标为负数")

        position = (record["file"], row, col)
        if position in seen_positions:
            raise ValueError(f"第 {index} 行重复指向图块：{position}")

        seen_positions.add(position)
        counts[(split, label)] += 1

    return counts


def _inspect_tiles(
    records: list[dict[str, str]], tiles_npy: Path
) -> tuple[Counter, dict, dict]:
    """检查像素，并以训练集分位数标记低清晰度候选。"""
    tiles = np.load(tiles_npy, mmap_mode="r", allow_pickle=False)
    expected_shape = (len(records), TILE_SIZE, TILE_SIZE)
    if tiles.shape != expected_shape or tiles.dtype != np.float32:
        raise ValueError(
            f"图块规格异常：实际 {tiles.shape}、{tiles.dtype}；"
            f"预期 {expected_shape}、float32"
        )

    quality = Counter()
    std_by_class = defaultdict(list)
    train_focus_by_class = defaultdict(list)
    focus_rows = []

    for index, record in enumerate(records):
        tile = tiles[index]
        if not np.isfinite(tile).all():
            quality["nonfinite"] += 1
        elif not np.any(tile != 0):
            quality["all_zero"] += 1
        elif tile.min() == tile.max():
            quality["constant"] += 1
        else:
            # 仅统计对比度，不据此自动删除低对比度细胞。
            std_by_class[record["label"]].append(float(tile.std()))
            # 只筛查 train/val，避免借测试图的质量分布调整训练方案。
            if record["split"] in {"train", "val"}:
                score = focus_proxy(tile)
                focus_rows.append((index, score))
                if record["split"] == "train":
                    train_focus_by_class[record["label"]].append(score)

    percentiles = {
        label: {
            "p01": float(np.percentile(values, 1)),
            "p50": float(np.percentile(values, 50)),
            "p99": float(np.percentile(values, 99)),
        }
        for label, values in std_by_class.items()
    }
    if set(train_focus_by_class) != set(SOURCE_CLASS_NAMES):
        raise ValueError("训练集缺少某个类别，无法按类别校准清晰度代理阈值")

    # 每类只使用训练集分布定阈值；低分只是待核查候选，不等于已证实模糊。
    thresholds = {
        label: float(np.percentile(scores, 1))
        for label, scores in train_focus_by_class.items()
    }
    candidate_counts = Counter()
    candidates = []
    for index, score in focus_rows:
        record = records[index]
        if score <= thresholds[record["label"]]:
            candidate_counts[(record["split"], record["label"])] += 1
            candidates.append((score, index))

    focus_screening = {
        "interpretation": "中央区域平滑后拉普拉斯方差；低值仅提示可能缺少细节，不能确认模糊或错标",
        "threshold_source": "每类训练图块清晰度代理分数的第 1 百分位",
        "thresholds": thresholds,
        "candidate_counts": {
            split: {label: candidate_counts[(split, label)] for label in SOURCE_CLASS_NAMES}
            for split in ("train", "val")
        },
        "lowest_candidates": [
            {
                "packed_index": index,
                "file": records[index]["file"],
                "row": int(records[index]["row"]),
                "col": int(records[index]["col"]),
                "score": score,
            }
            for score, index in sorted(candidates)[:25]
        ],
        "auto_excluded": 0,
    }
    return quality, percentiles, focus_screening


def audit_packed_data(split_csv: Path, tiles_npy: Path) -> dict:
    """组合清单校验和图块检查，返回不修改数据的审计报告。"""
    records = _read_records(split_csv)
    counts = _validate_records(records)
    quality, percentiles, focus_screening = _inspect_tiles(records, tiles_npy)

    return {
        "total_tiles": len(records),
        "counts": {
            split: {
                label: counts[(split, label)]
                for label in SOURCE_CLASS_NAMES
            }
            for split in ("train", "val", "test")
        },
        "quality_counts": {
            name: quality[name]
            for name in ("nonfinite", "all_zero", "constant")
        },
        "tile_std_percentiles": percentiles,
        "focus_screening": focus_screening,
    }


def write_audit_report(
    split_csv: Path, tiles_npy: Path, output_json: Path
) -> dict:
    """完成审计后写入 JSON；不覆盖已有报告。"""
    report = audit_packed_data(split_csv, tiles_npy)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    return report
