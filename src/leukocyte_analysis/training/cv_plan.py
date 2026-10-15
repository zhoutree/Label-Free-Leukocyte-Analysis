"""创建交叉验证计划，并核对分阶段运行的已保存结果。"""

import json
from pathlib import Path

from ..data.bbbc045_config import CV_GROUP_OVERRIDES, TEST_DONORS
from ..data.cv_split import DonorFold
from .cv_reports import fold_metrics, read_predictions


def build_cv_plan(split_csv: Path, tiles_npy: Path, folds: list[DonorFold],
                  seed: int, training: dict) -> dict:
    """保留现有 plan.json 字段和顺序，供续跑及事后对照。"""
    return {
        "method": "StratifiedGroupKFold",
        "group_overrides": CV_GROUP_OVERRIDES,
        "source_manifest": str(split_csv.resolve()),
        "packed_tiles": str(tiles_npy.resolve()),
        "excluded_donors": sorted(TEST_DONORS),
        "excluded_rule": "原测试受试者 CRF049 和 CRF066 不参与本次交叉验证；不是质量筛除",
        "selection_metric": "val_macro_f1",
        "seed": seed,
        "training": training,
        "folds": [
            {"number": fold.number, "train_donors": sorted(fold.train_donors),
             "val_donors": sorted(fold.val_donors)}
            for fold in folds
        ],
    }


def prepare_cv_run(output_dir: Path, plan: dict, folds: list[DonorFold], *,
                   continue_run: bool, label_smoothing: float) -> tuple[list[dict], list[dict]]:
    """新建运行目录，或从相同计划恢复已完成折。

    续跑前核对分折与训练配置，并返回已完成折的指标和预测；
    任何不一致都应阻止复用旧成绩。
    """
    if not continue_run:
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "plan.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return [], []

    saved_plan = json.loads((output_dir / "plan.json").read_text(encoding="utf-8"))
    # 旧分阶段运行尚无此字段；0 与关闭标签平滑等价，仍可安全接续。
    if "label_smoothing" not in saved_plan["training"] and label_smoothing == 0:
        del plan["training"]["label_smoothing"]
    if saved_plan != plan:
        raise ValueError("原运行的分折或训练配置与当前代码不一致，不能接续")
    saved_summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    if not saved_summary.get("partial") or saved_summary.get("plan") != plan:
        raise ValueError("只能接续同一计划下尚未完成的交叉验证")
    results = saved_summary["fold_results"]
    if [item["fold"] for item in results] != list(range(1, len(results) + 1)):
        raise ValueError("已完成的折不是从第一折开始连续排列")
    all_predictions = []
    for item in results:
        fold = folds[item["fold"] - 1]
        rows = read_predictions(
            output_dir / f"fold_{fold.number:02d}" / "validation_predictions.csv"
        )
        if (len(rows) != item["val_tiles"]
                or {row["donor"] for row in rows} != fold.val_donors
                or fold_metrics(rows) != item["metrics"]
                or not Path(item["checkpoint"]).is_file()):
            raise ValueError(f"第 {fold.number} 折的保存结果不完整")
        all_predictions.extend(rows)
    return results, all_predictions
