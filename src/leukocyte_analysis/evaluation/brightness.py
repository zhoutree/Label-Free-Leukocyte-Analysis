"""用已完成交叉验证的检查点比较固定亮度扰动下的折外表现。"""

import csv
import json
import tempfile
from collections import defaultdict
from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision.transforms.v2.functional import adjust_brightness

from ..data.bbbc045_config import CLASS_NAMES, TEST_DONORS
from ..data.dataset import LeukocyteDataset
from ..models.resnet import build_resnet18
from .metrics import classification_metrics
from .runner import check_checkpoint_head, load_classifier, predict


def _read_plan(run_dir: Path) -> dict:
    with (run_dir / "plan.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def _validate_plans(first: dict, second: dict) -> None:
    """只允许训练增强方式不同，避免把分折或训练设置差异当作亮度效果。"""
    for plan in (first, second):
        if set(plan["excluded_donors"]) != TEST_DONORS:
            raise ValueError("交叉验证必须排除 CRF049 和 CRF066")
        if not plan["folds"]:
            raise ValueError("交叉验证计划没有分折")
    fields = ("method", "source_manifest", "packed_tiles", "excluded_donors",
              "seed", "folds")
    if any(first[field] != second[field] for field in fields):
        raise ValueError("两次运行的数据、随机种子或受试者分折不一致")
    for plan in (first, second):
        seen = set()
        eligible = set(plan["folds"][0]["train_donors"]) | set(plan["folds"][0]["val_donors"])
        for fold in plan["folds"]:
            train, val = set(fold["train_donors"]), set(fold["val_donors"])
            if (not val or train & val or train | val != eligible
                    or (train | val) & TEST_DONORS or seen & val):
                raise ValueError("分折含重叠受试者或原测试受试者")
            seen.update(val)
        if seen != eligible:
            raise ValueError("分折未覆盖全部可用受试者")
    a = {key: value for key, value in first["training"].items() if key != "augmentation"}
    b = {key: value for key, value in second["training"].items() if key != "augmentation"}
    if a != b:
        raise ValueError("两次运行除增强方式外的训练超参数不一致")
    if (first["training"]["augmentation"] != "flips"
            or second["training"]["augmentation"] != "flips_photometric"):
        raise ValueError("应按顺序提供 flips 基线和 flips_photometric 增强运行")


def _measure(predictions: list[dict], run: str, factor: float) -> tuple[dict, list[dict], list[dict]]:
    """汇总总体、逐类召回率和逐受试者指标。"""
    def metrics(rows: list[dict]) -> dict:
        return classification_metrics(
            [row["true_index"] for row in rows],
            [row["predicted_index"] for row in rows], CLASS_NAMES,
        )

    overall = metrics(predictions)
    common = {"run": run, "brightness_factor": factor}
    summary = {**common, "samples": overall["samples"],
               "accuracy": overall["accuracy"], "macro_f1": overall["macro_f1"]}
    classes = [
        {**common, "class": name,
         "support": int(overall["classification_report"][name]["support"]),
         "recall": overall["classification_report"][name]["recall"]}
        for name in CLASS_NAMES
    ]
    grouped = defaultdict(list)
    for row in predictions:
        grouped[row["donor"]].append(row)
    donors = []
    for donor, rows in sorted(grouped.items()):
        result = metrics(rows)
        donors.append({**common, "donor": donor, "samples": result["samples"],
                       "accuracy": result["accuracy"], "macro_f1": result["macro_f1"]})
    return summary, classes, donors


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_plots(overall: list[dict], classes: list[dict], output_dir: Path) -> None:
    """用 matplotlib 绘制同一亮度倍率下两种训练方式的变化。"""
    runs = list(dict.fromkeys(row["run"] for row in overall))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for run in runs:
        rows = sorted((row for row in overall if row["run"] == run),
                      key=lambda row: row["brightness_factor"])
        for axis, metric in zip(axes, ("accuracy", "macro_f1")):
            axis.plot([row["brightness_factor"] for row in rows],
                      [row[metric] for row in rows], marker="o", label=run)
    for axis, metric in zip(axes, ("accuracy", "macro_f1")):
        axis.set(xlabel="Brightness multiplier", ylabel=metric, ylim=(0, 1.02))
        axis.grid(alpha=0.3)
    axes[0].legend()
    fig.savefig(output_dir / "overall.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for axis, name in zip(axes.flat, CLASS_NAMES):
        for run in runs:
            rows = sorted((row for row in classes if row["run"] == run
                           and row["class"] == name),
                          key=lambda row: row["brightness_factor"])
            axis.plot([row["brightness_factor"] for row in rows],
                      [row["recall"] for row in rows], marker="o", label=run)
        axis.set(title=name, xlabel="Brightness multiplier", ylabel="Recall", ylim=(0, 1.02))
        axis.grid(alpha=0.3)
    axes.flat[0].legend()
    fig.savefig(output_dir / "per_class_recall.png", dpi=160)
    plt.close(fig)


def compare_brightness(
    baseline_run: Path, augmented_run: Path, output_dir: Path, *,
    factors: tuple[float, ...] = (0.8, 1.0, 1.2), batch_size: int = 128,
) -> list[dict]:
    """只重跑各折验证集推理，不训练，也不读取原测试集。"""
    if batch_size < 1 or not factors or any(factor <= 0 for factor in factors):
        raise ValueError("batch-size 和所有亮度倍率必须大于 0")
    if len(set(factors)) != len(factors) or 1.0 not in factors:
        raise ValueError("亮度倍率不能重复，且必须包含未扰动的 1.0")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    plans = [_read_plan(run) for run in (baseline_run, augmented_run)]
    _validate_plans(*plans)
    tiles_npy = Path(plans[0]["packed_tiles"])
    if not tiles_npy.is_file():
        raise FileNotFoundError(tiles_npy)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    overall_rows, class_rows, donor_rows = [], [], []
    expected_indices = None
    for run_name, run_dir, plan in zip(("baseline", "photometric"),
                                       (baseline_run, augmented_run), plans):
        predictions_by_factor = {factor: [] for factor in factors}
        for fold in plan["folds"]:
            fold_dir = run_dir / f"fold_{fold['number']:02d}"
            checkpoint = fold_dir / "training/checkpoints/best.ckpt"
            check_checkpoint_head(checkpoint)
            model = load_classifier(
                checkpoint, build_resnet18(len(CLASS_NAMES), initialization="random")
            )
            for factor in factors:
                dataset = LeukocyteDataset(
                    fold_dir / "data/split.csv", tiles_npy,
                    fold_dir / "data/train_stats.json", "val",
                    raw_transform=(None if factor == 1.0 else partial(
                        adjust_brightness, brightness_factor=factor)),
                )
                if {row[2] for row in dataset.samples} != set(fold["val_donors"]):
                    raise ValueError("折内验证清单与 plan.json 的受试者不一致")
                print(f"{run_name} 第 {fold['number']} 折，亮度 ×{factor:g}", flush=True)
                predictions_by_factor[factor].extend(predict(model, dataset, batch_size, device))
            del model, dataset
        for factor, predictions in predictions_by_factor.items():
            indices = {row["packed_index"]: (row["donor"], row["true_index"])
                       for row in predictions}
            if (len(indices) != len(predictions) or not indices
                    or set(donor for donor, _ in indices.values()) & TEST_DONORS):
                raise ValueError("折外预测含重复图块、测试受试者或无样本")
            if expected_indices is None:
                expected_indices = indices
            elif indices != expected_indices:
                raise ValueError("两种运行或不同扰动的折外样本不一致")
            summary, classes, donors = _measure(predictions, run_name, factor)
            overall_rows.append(summary)
            class_rows.extend(classes)
            donor_rows.extend(donors)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".brightness-", dir=output_dir.parent) as temporary:
        staged = Path(temporary) / "report"
        staged.mkdir()
        _write_csv(staged / "overall.csv", overall_rows)
        _write_csv(staged / "per_class_recall.csv", class_rows)
        _write_csv(staged / "by_donor.csv", donor_rows)
        (staged / "settings.json").write_text(json.dumps({
            "baseline_run": str(baseline_run.resolve()),
            "augmented_run": str(augmented_run.resolve()),
            "brightness_factors": factors,
            "validation_only": True,
            "excluded_donors": sorted(TEST_DONORS),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        _save_plots(overall_rows, class_rows, staged)
        staged.rename(output_dir)
    return overall_rows
