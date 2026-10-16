"""BBBC045 四分类按受试者交叉验证入口，不接触原测试受试者。"""

import argparse
import os
from datetime import datetime
from pathlib import Path

from .training.cross_validation import continue_cross_validation, run_cross_validation
from .training.arguments import PROJECT_ROOT, add_training_arguments, validate_training_arguments


def parse_args() -> argparse.Namespace:
    """读取交叉验证设置；每折从头训练，不沿用上一折权重。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=3)
    add_training_arguments(parser)
    parser.add_argument("--smoke-test", action="store_true", help="仅检查第一折的两个训练和验证批次")
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument("--first-fold-only", action="store_true",
                        help="完整训练和评估第一折，保存计划以便稍后继续")
    stages.add_argument("--continue-run", type=Path,
                        help="沿用该运行目录的 plan.json，完成尚未训练的折")
    parser.add_argument("--relocate-inputs", action="store_true",
                        help="继续旧运行时改用当前 --split-csv 和 --tiles-npy 路径")
    args = parser.parse_args()
    if args.folds < 2:
        parser.error("folds 至少为 2")
    validate_training_arguments(parser, args)
    if args.smoke_test and (args.first_fold_only or args.continue_run):
        parser.error("冒烟测试不能与分阶段交叉验证同时使用")
    if args.continue_run and args.output_dir:
        parser.error("继续运行时目录由 --continue-run 指定，不再传 --output-dir")
    if args.relocate_inputs and not args.continue_run:
        parser.error("--relocate-inputs 只能与 --continue-run 同用")
    return args


def main() -> None:
    """生成独立运行目录并打印折外指标；不运行正式测试评估。"""
    args = parse_args()
    if args.init == "imagenet":
        os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "data/pretrained"))
    if args.continue_run:
        summary = continue_cross_validation(
            args.continue_run,
            split_csv=args.split_csv if args.relocate_inputs else None,
            tiles_npy=args.tiles_npy if args.relocate_inputs else None,
        )
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
