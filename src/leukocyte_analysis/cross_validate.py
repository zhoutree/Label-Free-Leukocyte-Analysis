"""BBBC045 四分类按受试者交叉验证入口，不接触原测试受试者。"""

import argparse
import os
from datetime import datetime
from pathlib import Path

from .training.cross_validation import run_cross_validation


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """读取交叉验证设置；每折从头训练，不沿用上一折权重。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--init", choices=("imagenet", "random"), default="imagenet")
    parser.add_argument("--augmentation", choices=("none", "flips"), default="flips")
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true", help="仅检查第一折的两个训练和验证批次")
    parser.add_argument("--split-csv", type=Path,
                        default=PROJECT_ROOT / "data/processed/bbbc045_split.csv")
    parser.add_argument("--tiles-npy", type=Path,
                        default=PROJECT_ROOT / "data/processed/bbbc045_tiles.npy")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if (args.folds < 2 or args.epochs < 1 or args.batch_size < 1 or args.lr <= 0
            or args.weight_decay < 0 or args.early_stopping_patience < 0):
        parser.error("folds 至少为 2；epochs、batch-size、lr 必须大于 0；衰减和早停耐心值不得为负")
    return args


def main() -> None:
    """生成独立运行目录并打印折外指标；不运行正式测试评估。"""
    args = parse_args()
    if args.init == "imagenet":
        os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "data/pretrained"))
    output_dir = args.output_dir or (
        PROJECT_ROOT / "runs"
        / f"cv_resnet18_{args.init}_{args.augmentation}_{datetime.now():%Y%m%d_%H%M%S}"
    )
    summary = run_cross_validation(
        args.split_csv, args.tiles_npy, output_dir,
        n_splits=args.folds, seed=args.seed, epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.lr,
        weight_decay=args.weight_decay, initialization=args.init,
        augmentation=args.augmentation,
        early_stopping_patience=args.early_stopping_patience,
        smoke_test=args.smoke_test,
    )
    if args.smoke_test:
        print("交叉验证冒烟测试通过；未保存模型权重。")
    else:
        metrics = summary["out_of_fold"]
        print(f"折外准确率：{metrics['accuracy']:.4f}；macro-F1：{metrics['macro_f1']:.4f}")
    print("结果目录：", output_dir)


if __name__ == "__main__":
    main()
