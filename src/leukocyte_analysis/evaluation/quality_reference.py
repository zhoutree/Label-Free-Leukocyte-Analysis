"""质量下限的校准、保存与读取；不依赖网络或 RMD 特征提取。"""

import csv
import json
import math
from pathlib import Path

import numpy as np

from ..data.bbbc045_config import CLASS_NAMES, model_label
from ..data.integrity import file_sha256
from ..data.split import read_reference_manifest
from ..paths import store_run_path
from ..reporting import new_report
from .cell_review import ReviewPolicy


def calibrate_focus(scores, labels, reject_rate: float = 0.01) -> dict:
    """限制每类验证图块的低清晰度拦截比例，返回统一的保守下限。

    校准集无正式质量标签，因此这是经验拦截比例，不是已验证的误判废率。
    取各类下分位工作点的最小值；推理时不需要真实类别或预测类别参与筛选。
    """
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels)
    if (not math.isfinite(reject_rate) or not 0 < reject_rate < 1
            or scores.ndim != 1 or scores.size == 0 or labels.shape != scores.shape
            or not np.isfinite(scores).all() or (scores < 0).any()
            or set(labels.tolist()) != set(range(len(CLASS_NAMES)))):
        raise ValueError("校准需要四类有限非负评分，且拦截目标须在 0 和 1 之间")
    groups = [np.sort(scores[labels == index]) for index in range(len(CLASS_NAMES))]
    # 严格小于阈值才拦截；顺序统计量保证并列分数和小样本不会超出预算。
    threshold = min(float(group[math.floor(reject_rate * len(group))]) for group in groups)
    return {
        "format_version": 1, "method": "normalized_tenengrad_v1",
        "class_names": list(CLASS_NAMES), "focus_min": threshold,
        "reject_rate_target_per_class": reject_rate, "samples": len(scores),
        "rejected": int((scores < threshold).sum()),
        "reject_rate_observed": float(np.mean(scores < threshold)),
        "per_class": [{"class": name, "samples": len(group),
                       "rejected": int((group < threshold).sum()),
                       "reject_rate_observed": float(np.mean(group < threshold))}
                      for name, group in zip(CLASS_NAMES, groups)],
        "limitations": "验证集低清晰度尾部的工程工作点，非真实不可用标签校准；"
                        "拦截比例不等于误判废率，不保证未知设备表现；仅标记、不删除原图。",
    }


def _validate_validation_rows(rows: list[dict], records: list[dict]) -> None:
    """逐项核对完整验证集索引、受试者及真实类别，拒绝混入或缺失样本。"""
    expected = {index: row for index, row in enumerate(records) if row["split"] == "val"}
    indices = [int(row["packed_index"]) for row in rows]
    if not expected or len(indices) != len(expected) or set(indices) != set(expected):
        raise ValueError("评分必须完整覆盖清单验证集，不能混入训练／测试图块、重复或缺失索引")
    for row in rows:
        original = expected[int(row["packed_index"])]
        label = CLASS_NAMES.index(model_label(original["label"]))
        if (row["donor"] != original["donor"] or int(row["true_index"]) != label
                or row.get("split", "val") != "val"):
            raise ValueError("评分的受试者、真实类别或划分与清单验证集不匹配")


def calibrate_quality(scores_csv: Path, output_dir: Path, reject_rate: float = 0.01,
                      *, split_csv: Path) -> dict:
    """用来源清单核验完整验证评分后校准，不加载图片、模型或改动原报告。"""
    records = read_reference_manifest(split_csv)
    with scores_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"packed_index", "donor", "true_index", "normalized_tenengrad"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("评分缺少来源索引、受试者、真实类别或清晰度列")
        rows = list(reader)
    _validate_validation_rows(rows, records)
    result = calibrate_focus([float(row["normalized_tenengrad"]) for row in rows],
                             [int(row["true_index"]) for row in rows], reject_rate)
    result["source"] = {"path": store_run_path(scores_csv, output_dir),
                        "sha256": file_sha256(scores_csv)}
    result["source_manifest"] = {"path": store_run_path(split_csv, output_dir),
                                 "sha256": file_sha256(split_csv)}
    with new_report(output_dir) as stage:
        (stage / "quality_threshold.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result


def load_quality_threshold(path: Path) -> float:
    """恢复绑定于当前清晰度算法的工作点，拒绝不兼容或无效的阈值。"""
    result = json.loads(path.read_text(encoding="utf-8"))
    if (result["format_version"] != 1 or result["method"] != "normalized_tenengrad_v1"
            or result["class_names"] != list(CLASS_NAMES)):
        raise ValueError("质量参考的算法或类别不匹配")
    threshold = float(result["focus_min"])
    ReviewPolicy(0, focus_min=threshold)
    return threshold
