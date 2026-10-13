"""BBBC045 四分类按受试者交叉验证入口，不接触原测试受试者。"""

import argparse
import math
import os
from datetime import datetime
from pathlib import Path

from .training.cross_validation import continue_cross_validation, run_cross_validation
from .training.experiment import EARLY_STOP_MIN_DELTA
from .training.loaders import AUGMENTATION_MODES


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """读取交叉验证设置；每折从头训练，不沿用上一折权重。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", choices=("none", "plateau", "cosine"), default="plateau")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--init", choices=("imagenet", "random"), default="imagenet")
    parser.add_argument("--augmentation", choices=AUGMENTATION_MODES,
                        default="flips_photometric_geometric")
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--early-stopping-min-delta", type=float,
                        default=EARLY_STOP_MIN_DELTA)
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="训练加权交叉熵的标签平滑系数；0 表示关闭")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true", help="仅检查第一折的两个训练和验证批次")
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument("--first-fold-only", action="store_true",
                        help="完整训练和评估第一折，保存计划以便稍后继续")
    stages.add_argument("--continue-run", type=Path,
                        help="沿用该运行目录的 plan.json，完成尚未训练的折")
    parser.add_argument("--split-csv", type=Path,
                        default=PROJECT_ROOT / "data/processed/bbbc045_split.csv")
    parser.add_argument("--tiles-npy", type=Path,
                        default=PROJECT_ROOT / "data/processed/bbbc045_tiles.npy")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if (args.folds < 2 or args.epochs < 1 or args.batch_size < 1 or args.lr <= 0
            or args.weight_decay < 0 or args.early_stopping_patience < 0):
        parser.error("folds 至少为 2；epochs、batch-size、lr 必须大于 0；衰减和早停耐心值不得为负")
    if (not math.isfinite(args.label_smoothing) or not 0 <= args.label_smoothing < 1
            or not math.isfinite(args.early_stopping_min_delta)
            or args.early_stopping_min_delta < 0):
        parser.error("标签平滑系数必须在 [0, 1) 内；早停最小改善量须为非负有限数")
    if args.smoke_test and (args.first_fold_only or args.continue_run):
        parser.error("冒烟测试不能与分阶段交叉验证同时使用")
    if args.continue_run and args.output_dir:
        parser.error("继续运行时目录由 --continue-run 指定，不再传 --output-dir")
    return args


def main() -> None:
    """生成独立运行目录并打印折外指标；不运行正式测试评估。"""
    args = parse_args()
    if args.init == "imagenet":
        os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "data/pretrained"))
    if args.continue_run:
        summary = continue_cross_validation(args.continue_run)
        metrics = summary["out_of_fold"]
        print(f"折外准确率：{metrics['accuracy']:.4f}；macro-F1：{metrics['macro_f1']:.4f}")
        print("结果目录：", args.continue_run)
        return
    output_dir = args.output_dir or (
        PROJECT_ROOT / "runs"
        / (f"cv_resnet18_{args.init}_{args.augmentation}_{args.lr_scheduler}"
           f"{'_ls' + format(args.label_smoothing, 'g') if args.label_smoothing else ''}"
           f"_{datetime.now():%Y%m%d_%H%M%S}")
    )
    summary = run_cross_validation(
        args.split_csv, args.tiles_npy, output_dir,
        n_splits=args.folds, seed=args.seed, epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.lr,
        weight_decay=args.weight_decay, initialization=args.init,
        augmentation=args.augmentation,
        lr_scheduler=args.lr_scheduler,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        label_smoothing=args.label_smoothing,
        smoke_test=args.smoke_test,
        first_fold_only=args.first_fold_only,
    )
    if args.smoke_test:
        print("交叉验证冒烟测试通过；未保存模型权重。")
    elif summary.get("partial"):
        metrics = summary["fold_results"][0]["metrics"]
        print(f"第 1 折验证准确率：{metrics['accuracy']:.4f}；macro-F1：{metrics['macro_f1']:.4f}")
        print("只完成第 1 折；这不是三折折外成绩。")
    else:
        metrics = summary["out_of_fold"]
        print(f"折外准确率：{metrics['accuracy']:.4f}；macro-F1：{metrics['macro_f1']:.4f}")
    print("结果目录：", output_dir)


if __name__ == "__main__":
    main()
