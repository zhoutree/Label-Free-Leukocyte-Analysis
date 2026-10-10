"""BBBC045 明场分类训练入口：读取参数并组装训练组件。"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import lightning as L
import torch

from .data.bbbc045_config import CLASS_NAMES
from .models.resnet import build_resnet18
from .training.experiment import run_training
from .training.loaders import make_loaders


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """读取命令行参数，并检查训练配置是否合法。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW 学习率；ImageNet 微调基线默认 0.0001")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW 权重衰减；不是 L1 正则化")
    parser.add_argument("--init", choices=("imagenet", "random"), default="imagenet", help="模型权重初始化方式")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true", help="只跑两批，不保存权重")
    parser.add_argument("--split-csv", type=Path, default=PROJECT_ROOT / "data/processed/bbbc045_split.csv")
    parser.add_argument("--tiles-npy", type=Path, default=PROJECT_ROOT / "data/processed/bbbc045_tiles.npy")
    parser.add_argument("--stats-json", type=Path, default=PROJECT_ROOT / "data/processed/bbbc045_train_stats.json")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.weight_decay < 0:
        parser.error("epochs、batch-size 和 lr 必须大于 0；weight-decay 不得小于 0")
    return args


def main() -> None:
    """连接数据、模型、优化器与训练流程；不读取测试集。"""
    args = parse_args()
    L.seed_everything(args.seed, workers=True)
    # 优先使用项目内的官方权重缓存；用户显式设置 TORCH_HOME 时尊重其选择。
    if args.init == "imagenet":
        os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "data/pretrained"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("设备：", device)
    train_loader, val_loader, weights = make_loaders(
        args.split_csv, args.tiles_npy, args.stats_json,
        args.batch_size, args.seed, device,
    )
    print("初始化：", args.init, "AdamW weight_decay：", args.weight_decay)
    model = build_resnet18(len(CLASS_NAMES), initialization=args.init)
    output_dir = args.output_dir or PROJECT_ROOT / "runs" / f"resnet18_{args.init}_{datetime.now():%Y%m%d_%H%M%S}"
    run_training(
        model, train_loader, val_loader, weights, device,
        args.epochs, output_dir, args.lr, args.weight_decay, args.smoke_test,
    )


if __name__ == "__main__":
    main()
