"""按受试者交叉验证四分类模型，始终排除原测试受试者。"""

import json
from pathlib import Path

import lightning as L
import torch
from ..data.bbbc045_config import (
    CLASS_NAMES, TEST_DONORS, TRAIN_DONORS, VAL_DONORS,
)
from ..data.cv_split import build_donor_folds, write_fold_manifest
from ..data.integrity import validate_bundle
from ..data.split import read_manifest
from ..data.stats import write_train_stats
from ..evaluation.inference import load_classifier, predict
from ..models.resnet import build_resnet18
from .cv_plan import build_cv_plan, prepare_cv_run
from .cv_reports import fold_metrics, summarize_predictions, write_predictions
from .experiment import COSINE_MIN_EPOCHS, EARLY_STOP_MIN_DELTA, run_training
from .loaders import make_loaders
from .module import (
    COSINE_MIN_LR_RATIO, COSINE_WARMUP_EPOCHS, COSINE_WARMUP_START_FACTOR,
    PLATEAU_COOLDOWN, PLATEAU_FACTOR, PLATEAU_MIN_LR_RATIO, PLATEAU_PATIENCE,
)
from ..paths import resolve_run_path, store_run_path


def run_cross_validation(
    split_csv: Path, tiles_npy: Path, output_dir: Path, *,
    n_splits: int, seed: int, epochs: int, batch_size: int,
    learning_rate: float, weight_decay: float, initialization: str,
    augmentation: str, early_stopping_patience: int, smoke_test: bool = False,
    lr_scheduler: str = "none",
    first_fold_only: bool = False,
    continue_run: bool = False,
    label_smoothing: float = 0.0,
    early_stopping_min_delta: float = EARLY_STOP_MIN_DELTA,
    relocated_inputs: bool = False,
) -> dict:
    """在固定留出的测试受试者之外训练并汇总逐折验证预测。

    每折独立计算训练像素统计量、初始化模型并保存最佳权重。
    ``first_fold_only`` 写入可接续的部分报告；``continue_run`` 先校验
    原计划再跳过已完成折；``smoke_test`` 只验证流程而不生成成绩。
    ``relocated_inputs`` 仅供用户显式更换旧计划中的数据路径。
    """
    output_dir = output_dir.resolve()
    if smoke_test and (first_fold_only or continue_run):
        raise ValueError("冒烟测试不能与分阶段交叉验证同时使用")
    if first_fold_only and continue_run:
        raise ValueError("首次只跑第一折与接续剩余折不能同时使用")
    fields, records = read_manifest(split_csv)
    expected_donors = TRAIN_DONORS | VAL_DONORS | TEST_DONORS
    if {row["donor"] for row in records} != expected_donors:
        raise ValueError("受试者名单与 BBBC045 配置不一致")
    if any((row["donor"] in TEST_DONORS) != (row["split"] == "test") for row in records):
        raise ValueError("原测试受试者与清单划分不一致")
    if not tiles_npy.is_file():
        raise FileNotFoundError(tiles_npy)
    if continue_run and relocated_inputs:
        # 核对完整清单、已保存折及其训练统计量绑定的图块包。
        saved_fold_csv = output_dir / "fold_01" / "data" / "split.csv"
        saved_stats_json = output_dir / "fold_01" / "data" / "train_stats.json"
        _, saved_rows = read_manifest(saved_fold_csv)
        if len(records) != len(saved_rows) or any(
            {key: value for key, value in row.items() if key != "split"}
            != {key: value for key, value in saved.items() if key != "split"}
            for row, saved in zip(records, saved_rows)
        ):
            raise ValueError("迁移后的清单与原运行保存的图块记录不一致")
        saved_stats = json.loads(saved_stats_json.read_text(encoding="utf-8"))
        validate_bundle(saved_fold_csv, tiles_npy, saved_stats, len(saved_rows))
    folds = build_donor_folds(records, n_splits, seed)
    training = {
        "epochs": epochs, "batch_size": batch_size,
        "learning_rate": learning_rate, "weight_decay": weight_decay,
        "initialization": initialization, "augmentation": augmentation,
        "label_smoothing": label_smoothing,
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
        "early_stopping_min_delta": early_stopping_min_delta,
    }
    plan = build_cv_plan(split_csv, tiles_npy, folds, seed, training, output_dir)
    if continue_run:
        saved_plan = json.loads((output_dir / "plan.json").read_text(encoding="utf-8"))
        for field, path in (("source_manifest", split_csv), ("packed_tiles", tiles_npy)):
            if resolve_run_path(output_dir, saved_plan[field]) == path.resolve():
                # 未迁移的旧版绝对路径仍按原格式比较，不改写历史计划。
                plan[field] = saved_plan[field]
    results, all_predictions = prepare_cv_run(
        output_dir, plan, folds, continue_run=continue_run,
        label_smoothing=label_smoothing,
        allow_relocated_inputs=relocated_inputs,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_folds = folds[:1] if smoke_test or first_fold_only else folds
    for fold in selected_folds[len(results):]:
        print(f"第 {fold.number}/{n_splits} 折；验证受试者：{', '.join(sorted(fold.val_donors))}")
        fold_dir = output_dir / f"fold_{fold.number:02d}"
        if fold_dir.exists():
            raise FileExistsError(f"第 {fold.number} 折已有未登记的文件，避免覆盖：{fold_dir}")
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
            early_stopping_min_delta=early_stopping_min_delta,
            label_smoothing=label_smoothing,
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
        write_predictions(fold_dir / "validation_predictions.csv", predictions)
        results.append({
            "fold": fold.number, "train_tiles": counts["train"],
            "val_tiles": counts["val"], "val_donors": sorted(fold.val_donors),
            "checkpoint": store_run_path(checkpoint, output_dir),
            "metrics": fold_metrics(predictions),
        })
        all_predictions.extend(predictions)
        if fold.number < n_splits:
            (output_dir / "summary.json").write_text(json.dumps({
                "smoke_test": False, "partial": True,
                "completed_folds": len(results), "plan": plan,
                "fold_results": results,
            }, ensure_ascii=False, indent=2), encoding="utf-8")

    if smoke_test:
        summary = {"smoke_test": True, "completed_folds": 1, "plan": plan}
    elif first_fold_only:
        summary = {"smoke_test": False, "partial": True,
                   "completed_folds": len(results), "plan": plan,
                   "fold_results": results}
    else:
        summary = summarize_predictions(plan, results, all_predictions, records)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if continue_run and relocated_inputs:
        (output_dir / "plan.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return summary


def continue_cross_validation(output_dir: Path, *, split_csv: Path | None = None,
                              tiles_npy: Path | None = None) -> dict:
    """按原 ``plan.json`` 的数据与超参数补齐剩余折。

    已完成折的权重和预测保持不变，返回更新后的折外汇总。
    迁移旧运行后可显式传入原清单和图块在新位置的路径。
    """
    plan = json.loads((output_dir / "plan.json").read_text(encoding="utf-8"))
    settings = plan["training"]
    return run_cross_validation(
        resolve_run_path(output_dir, plan["source_manifest"], split_csv),
        resolve_run_path(output_dir, plan["packed_tiles"], tiles_npy), output_dir,
        n_splits=len(plan["folds"]), seed=plan["seed"],
        epochs=settings["epochs"], batch_size=settings["batch_size"],
        learning_rate=settings["learning_rate"], weight_decay=settings["weight_decay"],
        initialization=settings["initialization"], augmentation=settings["augmentation"],
        early_stopping_patience=settings["early_stopping_patience"],
        lr_scheduler=settings["lr_scheduler"], continue_run=True,
        label_smoothing=settings.get("label_smoothing", 0.0),
        early_stopping_min_delta=settings.get("early_stopping_min_delta", 0.0),
        relocated_inputs=split_csv is not None or tiles_npy is not None,
    )
