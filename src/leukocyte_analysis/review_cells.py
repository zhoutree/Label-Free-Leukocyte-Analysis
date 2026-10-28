"""拟合已知细胞特征参考，或对原始灰度图块执行质量与待定检查。"""

import argparse
from pathlib import Path

from .evaluation.cell_review import ReviewPolicy
from .evaluation.novelty import known_threshold
from .evaluation.quality_reference import calibrate_quality
from .evaluation.review_runner import fit_reference, review_tiles
from .paths import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    """拟合、评分校准和应用分开；质量参考显式传入，不静默启用判废。"""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="训练侧拟合特征统计，验证侧校准已知误拒工作点")
    predict = commands.add_parser("predict", help="复核 N×55×55 float32 原始灰度图块包")
    quality = commands.add_parser("calibrate-quality", help="复用验证评分校准保守质量下限，无需推理")
    quality.add_argument("--validation-scores", type=Path, required=True)
    quality.add_argument("--split-csv", type=Path,
                         default=PROJECT_ROOT / "data/processed/bbbc045_split.csv",
                         help="评分来源的完整划分清单，用于核对验证集索引、标签与受试者")
    quality.add_argument("--output-dir", type=Path, required=True, help="必须是新目录")
    quality.add_argument("--quality-reject-rate", type=float, default=0.01,
                         help="每类验证图块的最大经验拦截比例；不是实测误判废率")
    for command in (fit, predict):
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True, help="必须是新目录")
        command.add_argument("--batch-size", type=int, default=128)
        command.add_argument("--tiles-npy", type=Path,
                             required=command is predict,
                             default=PROJECT_ROOT / "data/processed/bbbc045_tiles.npy"
                             if command is fit else None)
    fit.add_argument("--split-csv", type=Path,
                     default=PROJECT_ROOT / "data/processed/bbbc045_split.csv")
    fit.add_argument("--stats-json", type=Path,
                     default=PROJECT_ROOT / "data/processed/bbbc045_train_stats.json")
    fit.add_argument("--known-reject-rate", type=float, required=True,
                     help="已知类验证误拒目标，如 0.01；不是未知检出率")
    predict.add_argument("--reference", type=Path, required=True, help="fit 输出的 rmd.pt")
    focus = predict.add_mutually_exclusive_group()
    focus.add_argument("--quality-reference", type=Path, help="calibrate-quality 输出的 JSON")
    focus.add_argument("--focus-min", type=float,
                         help="人工核定的严重模糊下限，使用 normalized_tenengrad")
    predict.add_argument("--focus-review-min", type=float, help="清晰度存疑下限")
    args = parser.parse_args()
    try:
        if args.command == "calibrate-quality":
            known_threshold([0.0], args.quality_reject_rate)
            return args
        if args.batch_size < 1:
            raise ValueError("batch-size 必须大于 0")
        if args.command == "fit":
            known_threshold([0.0], args.known_reject_rate)
        else:
            ReviewPolicy(0.0, args.focus_min, args.focus_review_min)
    except ValueError as error:
        parser.error(str(error))
    return args


def main() -> None:
    """只编排独立模块；拟合不更新分类权重，应用不需要真实标签。"""
    args = parse_args()
    if args.command == "calibrate-quality":
        summary = calibrate_quality(args.validation_scores, args.output_dir,
                                    args.quality_reject_rate, split_csv=args.split_csv)
        print(f"清晰度下限：{summary['focus_min']:.8g}；"
              f"验证拦截比例：{summary['reject_rate_observed']:.2%}（非实测误判废率）。")
    elif args.command == "fit":
        summary = fit_reference(args.checkpoint, args.split_csv, args.tiles_npy,
                                args.stats_json, args.output_dir, args.known_reject_rate,
                                args.batch_size)
        print(f"RMD 工作点：{summary['ood_threshold']:.6g}；"
              f"验证已知误拒比例：{summary['known_reject_rate_observed']:.2%}。")
    else:
        summary = review_tiles(args.checkpoint, args.reference, args.tiles_npy, args.output_dir,
                               batch_size=args.batch_size, focus_min=args.focus_min,
                               focus_review_min=args.focus_review_min,
                               quality_reference=args.quality_reference)
        print("处理状态：", summary["decisions"])
    print("报告目录：", args.output_dir.resolve())


if __name__ == "__main__":
    main()
