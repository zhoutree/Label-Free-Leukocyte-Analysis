"""BBBC045 明场分类训练入口：读取参数并组装训练组件。"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import lightning as L
import torch

from .data.bbbc045_config import CLASS_NAMES, CV_GROUP_OVERRIDES
from .data.split import read_manifest, summarize_split_groups, validate_fixed_splits
from .models.resnet import build_resnet18
from .paths import PROJECT_ROOT
from .training.arguments import (DEFAULT_TRAIN_EPOCHS,
                                 add_training_arguments, validate_training_arguments)
from .training.experiment import run_training
from .training.loaders import make_loaders


def parse_args() -> argparse.Namespace:
    """读取命令行参数，并检查训练配置是否合法。"""
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser, default_epochs=DEFAULT_TRAIN_EPOCHS)
    parser.add_argument("--smoke-test", action="store_true", help="只跑两批，不保存权重")
    parser.add_argument("--stats-json", type=Path, default=PROJECT_ROOT / "data/processed/bbbc045_train_stats.json")
    args = parser.parse_args()
    validate_training_arguments(parser, args)
    return args


def main() -> None:
    """连接数据、模型、优化器与训练流程；不读取测试集。"""
    args = parse_args()
    _, records = read_manifest(args.split_csv)
    validate_fixed_splits(records)
    split_groups = summarize_split_groups(records)
    print("CRF101 同组规则：", CV_GROUP_OVERRIDES)
    L.seed_everything(args.seed, workers=True)
    # 优先使用项目内的官方权重缓存；用户显式设置 TORCH_HOME 时尊重其选择。
    if args.init == "imagenet":
        os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "data/pretrained"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("设备：", device)
    train_loader, val_loader, weights = make_loaders(
        args.split_csv, args.tiles_npy, args.stats_json,
        args.batch_size, args.seed, device, args.augmentation,
    )
    print("四分类：", CLASS_NAMES, "初始化：", args.init, "AdamW weight_decay：", args.weight_decay,
          "训练增强：", args.augmentation, "学习率调度：", args.lr_scheduler,
          "早停耐心值：", args.early_stopping_patience,
          "标签平滑：", args.label_smoothing)
    model = build_resnet18(len(CLASS_NAMES), initialization=args.init)
    output_dir = args.output_dir or PROJECT_ROOT / "runs" / (
        f"resnet18_four_{args.init}_{args.augmentation}_{args.lr_scheduler}"
        f"{'_ls' + format(args.label_smoothing, 'g') if args.label_smoothing else ''}_"
        f"{datetime.now():%Y%m%d_%H%M%S}"
    )
    run_training(
        model, train_loader, val_loader, weights, device,
        args.epochs, output_dir, args.lr, args.weight_decay, args.smoke_test,
        args.early_stopping_patience, lr_scheduler=args.lr_scheduler,
        early_stopping_min_delta=args.early_stopping_min_delta,
        label_smoothing=args.label_smoothing,
        data_split_record={
            "source_manifest": str(args.split_csv.resolve()),
            "group_overrides": CV_GROUP_OVERRIDES,
            "groups": split_groups,
        },
    )


if __name__ == "__main__":
    main()
