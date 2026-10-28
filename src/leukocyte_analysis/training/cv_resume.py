"""只延长一个交叉验证折，并复用其余折的原始预测。"""

import json
import shutil
from pathlib import Path

import lightning as L
import torch

from ..data.bbbc045_config import CLASS_NAMES, TEST_DONORS
from ..data.cv_split import build_donor_folds
from ..data.split import read_manifest
from ..evaluation.inference import load_classifier, predict
from ..models.resnet import build_resnet18
from .cv_reports import fold_metrics, read_predictions, summarize_predictions, write_predictions
from .experiment import run_training
from .loaders import make_loaders
from ..paths import resolve_run_path, store_run_path


def resume_cv_fold(source_dir: Path, output_dir: Path, *, fold_number: int,
                   epochs: int, split_csv: Path | None = None,
                   tiles_npy: Path | None = None) -> dict:
    """从指定折的 ``last.ckpt`` 续训，并与原最佳权重比较。

    原运行保持只读；新目录保存续训日志及选择结果。只有当续训后的
    macro-F1 高于原结果时才替换该折预测，其他折始终复用原预测。
    ``epochs`` 是续训后的总轮数上限，而非额外轮数。
    迁移旧运行时可覆盖计划中失效的清单和图块路径。
    """
    source_dir = source_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"续训结果目录已存在：{output_dir}")
    if output_dir.resolve().is_relative_to(source_dir):
        raise ValueError("续训结果目录不能放在原运行目录内部")
    source = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    if source.get("smoke_test"):
        raise ValueError("不能续训冒烟测试")
    plan = source["plan"]
    settings = plan["training"]
    if plan["excluded_donors"] != sorted(TEST_DONORS):
        raise ValueError("原运行的测试受试者排除规则与当前项目不一致")
    if epochs <= settings["epochs"]:
        raise ValueError("新的最大轮数必须大于原运行的最大轮数")
    fold_plan = next((item for item in plan["folds"]
                      if item["number"] == fold_number), None)
    if fold_plan is None:
        raise ValueError(f"原运行没有第 {fold_number} 折")
    if len(source["fold_results"]) != len(plan["folds"]):
        raise ValueError("原运行尚未完成全部折")

    source_manifest = resolve_run_path(source_dir, plan["source_manifest"], split_csv)
    packed_tiles = resolve_run_path(source_dir, plan["packed_tiles"], tiles_npy)
    if not packed_tiles.is_file():
        raise FileNotFoundError(packed_tiles)
    _, records = read_manifest(source_manifest)
    folds = build_donor_folds(records, len(plan["folds"]), plan["seed"])
    fold = folds[fold_number - 1]
    if (sorted(fold.train_donors) != fold_plan["train_donors"]
            or sorted(fold.val_donors) != fold_plan["val_donors"]):
        raise ValueError("当前清单生成的受试者分折与原运行不一致")

    fold_dir = source_dir / f"fold_{fold_number:02d}"
    training_dir = fold_dir / "training"
    last_checkpoint = training_dir / "checkpoints" / "last.ckpt"
    old_checkpoint = training_dir / "checkpoints" / "best.ckpt"
    for path in (last_checkpoint, old_checkpoint, fold_dir / "data" / "split.csv",
                 fold_dir / "data" / "train_stats.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    _, fold_records = read_manifest(fold_dir / "data" / "split.csv")
    if len(fold_records) != len(records) or any(
        {key: value for key, value in original.items() if key != "split"}
        != {key: value for key, value in saved.items() if key != "split"}
        or saved["split"] != (
            "train" if original["donor"] in fold.train_donors else
            "val" if original["donor"] in fold.val_donors else "excluded"
        )
        for original, saved in zip(records, fold_records)
    ):
        raise ValueError("原始清单与该折保存的划分不一致，不能安全续训")

    # 新目录存续训日志；复制旧错分记录，保证 frequency.csv 统计完整轮次。
    output_dir.mkdir(parents=True, exist_ok=False)
    new_training_dir = output_dir / f"fold_{fold_number:02d}" / "training"
    old_errors = training_dir / "validation_errors"
    if old_errors.is_dir():
        shutil.copytree(old_errors, new_training_dir / "validation_errors")
    (output_dir / "continuation_plan.json").write_text(json.dumps({
        "source_run": str(source_dir), "fold": fold_number,
        "resume_checkpoint": str(last_checkpoint), "max_epochs": epochs,
        "original_max_epochs": settings["epochs"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L.seed_everything(plan["seed"], workers=True)
    train_loader, val_loader, weights = make_loaders(
        fold_dir / "data" / "split.csv", packed_tiles,
        fold_dir / "data" / "train_stats.json", settings["batch_size"],
        plan["seed"], device, settings["augmentation"],
    )
    # 权重和优化器由检查点恢复；调度周期沿用原计划，避免续训时重启学习率曲线。
    run_training(
        build_resnet18(len(CLASS_NAMES), initialization="random"),
        train_loader, val_loader, weights, device, epochs, new_training_dir,
        settings["learning_rate"], settings["weight_decay"],
        early_stopping_patience=settings["early_stopping_patience"],
        lr_scheduler=settings.get("lr_scheduler", "none"),
        resume_checkpoint=last_checkpoint,
        schedule_epochs=settings["epochs"],
        early_stopping_min_delta=settings.get("early_stopping_min_delta", 0.0),
        label_smoothing=settings.get("label_smoothing", 0.0),
    )

    original_rows = read_predictions(fold_dir / "validation_predictions.csv")
    original_metrics = fold_metrics(original_rows)
    new_checkpoint = new_training_dir / "checkpoints" / "best.ckpt"
    if new_checkpoint.is_file():
        model = load_classifier(
            new_checkpoint, build_resnet18(len(CLASS_NAMES), initialization="random")
        )
        new_rows = predict(model, val_loader.dataset, settings["batch_size"], device)
        new_metrics = fold_metrics(new_rows)
    else:
        new_rows, new_metrics = [], None

    improved = new_metrics is not None and new_metrics["macro_f1"] > original_metrics["macro_f1"]
    selected_rows = new_rows if improved else original_rows
    selected_checkpoint = new_checkpoint if improved else old_checkpoint
    fold_output = output_dir / f"fold_{fold_number:02d}" / "validation_predictions.csv"
    write_predictions(fold_output, selected_rows)

    results = []
    all_predictions = []
    for item in source["fold_results"]:
        number = item["fold"]
        rows = (selected_rows if number == fold_number else read_predictions(
            source_dir / f"fold_{number:02d}" / "validation_predictions.csv"
        ))
        expected_donors = next(set(part["val_donors"]) for part in plan["folds"]
                               if part["number"] == number)
        if {row["donor"] for row in rows} != expected_donors:
            raise ValueError(f"第 {number} 折的预测受试者与计划不一致")
        checkpoint = (selected_checkpoint if number == fold_number else
                      source_dir / f"fold_{number:02d}" / "training" / "checkpoints" / "best.ckpt")
        result = {**item, "checkpoint": store_run_path(checkpoint, output_dir)}
        if number == fold_number:
            result["metrics"] = fold_metrics(rows)
        results.append(result)
        all_predictions.extend(rows)

    plan = {**plan, "source_manifest": store_run_path(source_manifest, output_dir),
            "packed_tiles": store_run_path(packed_tiles, output_dir)}
    summary = summarize_predictions(plan, results, all_predictions, records)
    summary["continuation"] = {
        "source_run": str(source_dir), "fold": fold_number,
        "max_epochs": epochs, "selected": "extended" if improved else "original",
        "original_macro_f1": original_metrics["macro_f1"],
        "extended_macro_f1": None if new_metrics is None else new_metrics["macro_f1"],
        "note": "其余折复用原预测；原运行未修改。",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
