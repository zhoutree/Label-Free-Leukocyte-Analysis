"""按受试者交叉验证四分类模型，始终排除原测试受试者。"""

import csv
import json
from pathlib import Path

import lightning as L
import numpy as np
import torch
from ..data.bbbc045_config import (
    CLASS_NAMES, CV_GROUP_OVERRIDES, TEST_DONORS, TRAIN_DONORS, VAL_DONORS,
)
from ..data.cv_split import build_donor_folds, write_fold_manifest
from ..data.split import read_manifest
from ..data.stats import write_train_stats
from ..evaluation.metrics import classification_metrics
from ..evaluation.runner import load_classifier, predict
from ..models.resnet import build_resnet18
from .experiment import COSINE_MIN_EPOCHS, EARLY_STOP_MIN_DELTA, run_training
from .loaders import make_loaders
from .module import (
    COSINE_MIN_LR_RATIO, COSINE_WARMUP_EPOCHS, COSINE_WARMUP_START_FACTOR,
    PLATEAU_COOLDOWN, PLATEAU_FACTOR, PLATEAU_MIN_LR_RATIO, PLATEAU_PATIENCE,
)


def _metrics(rows: list[dict]) -> dict:
    """报告准确率、macro-F1 和每类指标，不用测试集选择配置。"""
    y_true = [int(row["true_index"]) for row in rows]
    y_pred = [int(row["predicted_index"]) for row in rows]
    metrics = classification_metrics(y_true, y_pred, CLASS_NAMES)
    return {
        "samples": metrics["samples"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "per_class": {name: metrics["classification_report"][name] for name in CLASS_NAMES},
    }


def _write_predictions(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_predictions(plan: dict, results: list[dict],
                          all_predictions: list[dict], records: list[dict]) -> dict:
    """检查折外覆盖范围，并汇总完整三折结果。"""
    eligible_donors = TRAIN_DONORS | VAL_DONORS
    if ({row["donor"] for row in all_predictions} != eligible_donors
            or len(all_predictions) != sum(row["donor"] in eligible_donors for row in records)
            or len({int(row["packed_index"]) for row in all_predictions}) != len(all_predictions)):
        raise ValueError("折外预测缺少受试者或重复图块")
    return {
        "smoke_test": False,
        "plan": plan,
        "fold_results": results,
        "out_of_fold": _metrics(all_predictions),
        "fold_macro_f1_mean": float(np.mean([row["metrics"]["macro_f1"] for row in results])),
        "fold_macro_f1_std": float(np.std([row["metrics"]["macro_f1"] for row in results])),
        "by_donor": {
            donor: _metrics([row for row in all_predictions if row["donor"] == donor])
            for donor in sorted(eligible_donors)
        },
    }


def run_cross_validation(
    split_csv: Path, tiles_npy: Path, output_dir: Path, *,
    n_splits: int, seed: int, epochs: int, batch_size: int,
    learning_rate: float, weight_decay: float, initialization: str,
    augmentation: str, early_stopping_patience: int, smoke_test: bool = False,
    lr_scheduler: str = "none",
) -> dict:
    """逐折重算统计量、重新初始化训练，并汇总折外预测。"""
    fields, records = read_manifest(split_csv)
    expected_donors = TRAIN_DONORS | VAL_DONORS | TEST_DONORS
    if {row["donor"] for row in records} != expected_donors:
        raise ValueError("受试者名单与 BBBC045 配置不一致")
    if any((row["donor"] in TEST_DONORS) != (row["split"] == "test") for row in records):
        raise ValueError("原测试受试者与清单划分不一致")
    if not tiles_npy.is_file():
        raise FileNotFoundError(tiles_npy)
    folds = build_donor_folds(records, n_splits, seed)
    output_dir.mkdir(parents=True, exist_ok=False)

    plan = {
        "method": "StratifiedGroupKFold",
        "group_overrides": CV_GROUP_OVERRIDES,
        "source_manifest": str(split_csv.resolve()),
        "packed_tiles": str(tiles_npy.resolve()),
        "excluded_donors": sorted(TEST_DONORS),
        "excluded_rule": "原测试受试者 CRF049 和 CRF066 不参与本次交叉验证；不是质量筛除",
        "selection_metric": "val_macro_f1",
        "seed": seed,
        "training": {
            "epochs": epochs, "batch_size": batch_size,
            "learning_rate": learning_rate, "weight_decay": weight_decay,
            "initialization": initialization, "augmentation": augmentation,
            "lr_scheduler": lr_scheduler,
            "plateau_settings": {
                "factor": PLATEAU_FACTOR, "patience": PLATEAU_PATIENCE,
                "cooldown": PLATEAU_COOLDOWN, "min_lr_ratio": PLATEAU_MIN_LR_RATIO,
            } if lr_scheduler == "plateau" else None,
            "cosine_settings": {
                "warmup_epochs": min(COSINE_WARMUP_EPOCHS, max(0, epochs - 2)),
                "warmup_start_factor": COSINE_WARMUP_START_FACTOR,
                "min_lr_ratio": COSINE_MIN_LR_RATIO,
                "min_epochs_before_early_stop": min(COSINE_MIN_EPOCHS, epochs),
            } if lr_scheduler == "cosine" else None,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": EARLY_STOP_MIN_DELTA,
        },
        "folds": [
            {"number": fold.number, "train_donors": sorted(fold.train_donors),
             "val_donors": sorted(fold.val_donors)}
            for fold in folds
        ],
    }
    (output_dir / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_predictions = []
    results = []
    for fold in folds[:1] if smoke_test else folds:
        print(f"第 {fold.number}/{n_splits} 折；验证受试者：{', '.join(sorted(fold.val_donors))}")
        fold_dir = output_dir / f"fold_{fold.number:02d}"
        data_dir = fold_dir / "data"
        fold_csv = data_dir / "split.csv"
        counts = write_fold_manifest(fields, records, fold, fold_csv)
        stats_json = data_dir / "train_stats.json"
        write_train_stats(fold_csv, tiles_npy, stats_json)

        # 各折的模型初始化、训练顺序与归一化统计相互独立。
        L.seed_everything(seed, workers=True)
        train_loader, val_loader, weights = make_loaders(
            fold_csv, tiles_npy, stats_json, batch_size, seed, device, augmentation
        )
        backbone = build_resnet18(len(CLASS_NAMES), initialization=initialization)
        training_dir = fold_dir / "training"
        run_training(
            backbone, train_loader, val_loader, weights, device, epochs,
            training_dir, learning_rate, weight_decay, smoke_test,
            early_stopping_patience, lr_scheduler=lr_scheduler,
        )
        if smoke_test:
            break

        checkpoint = training_dir / "checkpoints" / "best.ckpt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"第 {fold.number} 折没有最佳检查点：{checkpoint}")
        model = load_classifier(
            checkpoint, build_resnet18(len(CLASS_NAMES), initialization="random")
        )
        predictions = predict(model, val_loader.dataset, batch_size, device)
        if {row["donor"] for row in predictions} != fold.val_donors:
            raise ValueError("预测结果包含非本折验证受试者")
        _write_predictions(fold_dir / "validation_predictions.csv", predictions)
        results.append({
            "fold": fold.number, "train_tiles": counts["train"],
            "val_tiles": counts["val"], "val_donors": sorted(fold.val_donors),
            "checkpoint": str(checkpoint), "metrics": _metrics(predictions),
        })
        all_predictions.extend(predictions)

    if smoke_test:
        summary = {"smoke_test": True, "completed_folds": 1, "plan": plan}
    else:
        summary = summarize_predictions(plan, results, all_predictions, records)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
