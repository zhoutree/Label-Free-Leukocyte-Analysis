"""按受试者交叉验证四分类模型，始终排除原测试受试者。"""

import csv
import json
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import lightning as L
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold

from ..data.bbbc045_config import (
    CLASS_NAMES,
    TEST_DONORS,
    TRAIN_DONORS,
    VAL_DONORS,
    model_label,
)
from ..data.stats import write_train_stats
from ..evaluation.runner import load_classifier, predict
from ..models.resnet import build_resnet18
from .experiment import run_training
from .loaders import make_loaders


@dataclass(frozen=True)
class DonorFold:
    number: int
    train_donors: frozenset[str]
    val_donors: frozenset[str]


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """读取原划分清单；后续仅修改复制品中的 split 列。"""
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"file", "donor", "label", "row", "col", "split"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("划分清单缺少交叉验证所需的列")
        fields = list(reader.fieldnames)
        records = list(reader)
    if not records:
        raise ValueError("划分清单为空")
    return fields, records


def build_donor_folds(
    records: list[dict[str, str]], n_splits: int, seed: int,
    excluded_donors: frozenset[str] = TEST_DONORS,
) -> list[DonorFold]:
    """用 scikit-learn 分折，确保一个受试者只进入一折验证集。"""
    if n_splits < 2:
        raise ValueError("交叉验证至少需要 2 折")
    eligible = [row for row in records if row["donor"] not in excluded_donors]
    donors = {row["donor"] for row in eligible}
    if n_splits > len(donors):
        raise ValueError("折数不能超过可用受试者人数")

    labels = [model_label(row["label"]) for row in eligible]
    groups = [row["donor"] for row in eligible]
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    seen_validation = set()
    for number, (_train_indices, val_indices) in enumerate(
        splitter.split(np.zeros(len(eligible)), labels, groups), start=1
    ):
        val_donors = frozenset(groups[index] for index in val_indices)
        train_donors = frozenset(donors - val_donors)
        if not val_donors or not train_donors or seen_validation & val_donors:
            raise ValueError("受试者分折不完整或重复")
        seen_validation.update(val_donors)
        folds.append(DonorFold(number, train_donors, val_donors))
    if seen_validation != donors:
        raise ValueError("有受试者没有进入任何验证折")
    return folds


def write_fold_manifest(
    fields: list[str], records: list[dict[str, str]], fold: DonorFold,
    output_csv: Path,
) -> Counter:
    """保留全部原行和索引；测试受试者标为 excluded，不供数据集读取。"""
    if output_csv.exists():
        raise FileExistsError(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with tempfile.TemporaryDirectory(prefix=".fold-", dir=output_csv.parent) as temporary:
        staged = Path(temporary) / output_csv.name
        with staged.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for record in records:
                donor = record["donor"]
                if donor in fold.train_donors:
                    split = "train"
                elif donor in fold.val_donors:
                    split = "val"
                else:
                    split = "excluded"
                writer.writerow({**record, "split": split})
                counts[split] += 1
        staged.rename(output_csv)
    return counts


def _metrics(rows: list[dict]) -> dict:
    """报告准确率、macro-F1 和每类指标，不用测试集选择配置。"""
    y_true = [int(row["true_index"]) for row in rows]
    y_pred = [int(row["predicted_index"]) for row in rows]
    report = classification_report(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES))),
        target_names=list(CLASS_NAMES), output_dict=True, zero_division=0,
    )
    return {
        "samples": len(rows),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "per_class": {name: report[name] for name in CLASS_NAMES},
    }


def _write_predictions(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_cross_validation(
    split_csv: Path, tiles_npy: Path, output_dir: Path, *,
    n_splits: int, seed: int, epochs: int, batch_size: int,
    learning_rate: float, weight_decay: float, initialization: str,
    augmentation: str, early_stopping_patience: int, smoke_test: bool = False,
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
        "source_manifest": str(split_csv.resolve()),
        "packed_tiles": str(tiles_npy.resolve()),
        "excluded_donors": sorted(TEST_DONORS),
        "excluded_rule": "CRF049 和 CRF066 的全部图块（包括 CRF049/monocyte）不参与交叉验证",
        "selection_metric": "val_macro_f1",
        "seed": seed,
        "training": {
            "epochs": epochs, "batch_size": batch_size,
            "learning_rate": learning_rate, "weight_decay": weight_decay,
            "initialization": initialization, "augmentation": augmentation,
            "early_stopping_patience": early_stopping_patience,
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
            early_stopping_patience,
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
        eligible_donors = (TRAIN_DONORS | VAL_DONORS)
        if ({row["donor"] for row in all_predictions} != eligible_donors
                or len(all_predictions) != sum(row["donor"] in eligible_donors for row in records)
                or len({row["packed_index"] for row in all_predictions}) != len(all_predictions)):
            raise ValueError("折外预测缺少受试者或重复图块")
        summary = {
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
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
