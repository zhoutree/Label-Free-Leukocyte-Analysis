"""常规训练与交叉验证共用的命令行训练参数。"""

import argparse
import math
from pathlib import Path

from .loaders import AUGMENTATION_MODES


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRAIN_EPOCHS = 45
DEFAULT_CV_EPOCHS = 30


def add_training_arguments(parser: argparse.ArgumentParser, *,
                           default_epochs: int = DEFAULT_CV_EPOCHS) -> None:
    """加入共用超参数；常规训练和交叉验证可分别指定轮数上限。"""
    parser.add_argument("--epochs", type=int, default=default_epochs)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW 学习率；ImageNet 微调基线默认 0.0001")
    parser.add_argument("--lr-scheduler", choices=("none", "plateau", "cosine"), default="plateau",
                        help="默认按验证 macro-F1 停滞情况降低学习率；可选择余弦调度")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW 权重衰减；不是 L1 正则化")
    parser.add_argument("--init", choices=("imagenet", "random"), default="imagenet",
                        help="模型权重初始化方式")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augmentation", choices=AUGMENTATION_MODES,
                        default="flips_photometric_geometric",
                        help="训练集增强；默认翻转、亮度／对比度扰动和保守几何变换")
    parser.add_argument("--early-stopping-patience", type=int, default=15,
                        help="验证 macro-F1 连续多少轮未提高就停止；0 表示禁用")
    parser.add_argument("--early-stopping-min-delta", type=float,
                        default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05,
                        help="训练加权交叉熵的标签平滑系数；默认 0.05，0 表示关闭")
    parser.add_argument("--split-csv", type=Path,
                        default=PROJECT_ROOT / "data/processed/bbbc045_split.csv")
    parser.add_argument("--tiles-npy", type=Path,
                        default=PROJECT_ROOT / "data/processed/bbbc045_tiles.npy")
    parser.add_argument("--output-dir", type=Path)


def validate_training_arguments(parser: argparse.ArgumentParser,
                                args: argparse.Namespace) -> None:
    """拒绝两种入口共同不允许的训练超参数。"""
    if (args.epochs < 1 or args.batch_size < 1 or not math.isfinite(args.lr)
            or args.lr <= 0 or not math.isfinite(args.weight_decay)
            or args.weight_decay < 0 or args.early_stopping_patience < 0):
        parser.error("epochs、batch-size 和有限的 lr 必须大于 0；weight-decay 必须为非负有限数，early-stopping-patience 不得小于 0")
    if (not math.isfinite(args.label_smoothing) or not 0 <= args.label_smoothing < 1
            or not math.isfinite(args.early_stopping_min_delta)
            or args.early_stopping_min_delta < 0):
        parser.error("标签平滑系数必须在 [0, 1) 内；早停最小改善量须为非负有限数")
