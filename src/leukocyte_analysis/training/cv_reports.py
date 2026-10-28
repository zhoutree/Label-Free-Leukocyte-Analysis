"""交叉验证与单折续训共用的预测记录和折外指标。"""

import csv
from pathlib import Path

import numpy as np

from ..data.bbbc045_config import CLASS_NAMES, TRAIN_DONORS, VAL_DONORS
from ..evaluation.metrics import classification_metrics
from ..reporting import write_csv


def fold_metrics(rows: list[dict]) -> dict:
    """用统一的四分类口径汇总一组折外预测。"""
    y_true = [int(row["true_index"]) for row in rows]
    y_pred = [int(row["predicted_index"]) for row in rows]
    metrics = classification_metrics(y_true, y_pred, CLASS_NAMES)
    return {
        "samples": metrics["samples"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "per_class": {name: metrics["classification_report"][name] for name in CLASS_NAMES},
    }


def read_predictions(path: Path) -> list[dict]:
    """读取已保存的一折预测；CSV 数值列交由 fold_metrics 转换。"""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_predictions(path: Path, rows: list[dict]) -> None:
    """只创建新报告，拒绝覆盖已完成或正在核对的预测。"""
    write_csv(path, rows, tuple(rows[0]), overwrite=False)


def summarize_predictions(plan: dict, results: list[dict],
                          all_predictions: list[dict], records: list[dict]) -> dict:
    """核对折外数量、受试者覆盖和索引去重，并生成完整报告。

    按四类顺序计算总体与受试者指标；原保留测试受试者不计入折外成绩。
    """
    eligible_donors = TRAIN_DONORS | VAL_DONORS
    if ({row["donor"] for row in all_predictions} != eligible_donors
            or len(all_predictions) != sum(row["donor"] in eligible_donors for row in records)
            or len({int(row["packed_index"]) for row in all_predictions}) != len(all_predictions)):
        raise ValueError("折外预测缺少受试者或重复图块")
    return {
        "smoke_test": False,
        "plan": plan,
        "fold_results": results,
        "out_of_fold": fold_metrics(all_predictions),
        "fold_macro_f1_mean": float(np.mean([row["metrics"]["macro_f1"] for row in results])),
        "fold_macro_f1_std": float(np.std([row["metrics"]["macro_f1"] for row in results])),
        "by_donor": {
            donor: fold_metrics([row for row in all_predictions if row["donor"] == donor])
            for donor in sorted(eligible_donors)
        },
    }
