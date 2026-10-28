"""用已完成交叉验证的检查点比较固定亮度扰动下的折外表现。"""

import json
from collections import defaultdict
from functools import partial
from pathlib import Path

import torch
from torchvision.transforms.v2.functional import adjust_brightness

from ..data.bbbc045_config import CLASS_NAMES, TEST_DONORS
from ..data.dataset import LeukocyteDataset
from ..models.resnet import build_resnet18
from ..reporting import new_report, write_csv
from ..paths import resolve_run_path
from .metrics import classification_metrics
from .plots import save_brightness_comparison
from .inference import check_checkpoint_head, load_classifier, predict


def _read_plan(run_dir: Path) -> dict:
    """读取指定运行的分折与训练配置。"""
    with (run_dir / "plan.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def _validate_plans(first: dict, second: dict,
                    run_dirs: tuple[Path, Path] | None = None) -> None:
    """只允许训练增强方式不同，避免把分折或训练设置差异当作亮度效果。"""
    for plan in (first, second):
        if set(plan["excluded_donors"]) != TEST_DONORS:
            raise ValueError("交叉验证必须排除 CRF049 和 CRF066")
        if not plan["folds"]:
            raise ValueError("交叉验证计划没有分折")
    fields = ("method", "excluded_donors", "seed", "folds")
    if any(first[field] != second[field] for field in fields):
        raise ValueError("两次运行的数据、随机种子或受试者分折不一致")
    for field in ("source_manifest", "packed_tiles"):
        same_source = (first[field] == second[field] if run_dirs is None else
                       resolve_run_path(run_dirs[0], first[field])
                       == resolve_run_path(run_dirs[1], second[field]))
        if not same_source:
            raise ValueError("两次运行的数据路径不一致")
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
        """按统一类别顺序计算一组预测的分类指标。"""
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


def compare_brightness(
    baseline_run: Path, augmented_run: Path, output_dir: Path, *,
    factors: tuple[float, ...] = (0.8, 1.0, 1.2), batch_size: int = 128,
    tiles_npy: Path | None = None,
) -> list[dict]:
    """用两组固定权重比较相同折外图块的亮度扰动响应。

    两次运行除训练增强外必须具有相同数据、分折与超参数。
    各亮度倍率只用于验证集推理，原测试集不读取；结果写入全新
    报告目录，返回每组倍率的总体指标行。迁移旧运行时可覆盖图块路径。
    """
    if batch_size < 1 or not factors or any(factor <= 0 for factor in factors):
        raise ValueError("batch-size 和所有亮度倍率必须大于 0")
    if len(set(factors)) != len(factors) or 1.0 not in factors:
        raise ValueError("亮度倍率不能重复，且必须包含未扰动的 1.0")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    plans = [_read_plan(run) for run in (baseline_run, augmented_run)]
    _validate_plans(*plans, run_dirs=(baseline_run, augmented_run))
    packed_tiles = resolve_run_path(baseline_run, plans[0]["packed_tiles"], tiles_npy)
    if not packed_tiles.is_file():
        raise FileNotFoundError(packed_tiles)

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
                    fold_dir / "data/split.csv", packed_tiles,
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

    with new_report(output_dir, prefix=".brightness-") as staged:
        write_csv(staged / "overall.csv", overall_rows, tuple(overall_rows[0]))
        write_csv(staged / "per_class_recall.csv", class_rows, tuple(class_rows[0]))
        write_csv(staged / "by_donor.csv", donor_rows, tuple(donor_rows[0]))
        (staged / "settings.json").write_text(json.dumps({
            "baseline_run": str(baseline_run.resolve()),
            "augmented_run": str(augmented_run.resolve()),
            "brightness_factors": factors,
            "validation_only": True,
            "excluded_donors": sorted(TEST_DONORS),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        save_brightness_comparison(overall_rows, class_rows, staged, CLASS_NAMES)
    return overall_rows
