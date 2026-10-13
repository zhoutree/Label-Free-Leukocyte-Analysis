"""构建 BBBC045 训练/验证数据加载器和类别权重。"""

from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from ..data.bbbc045_config import CLASS_NAMES
from ..data.dataset import LeukocyteDataset


AUGMENTATION_MODES = ("none", "flips", "flips_photometric", "flips_photometric_geometric")


def make_loaders(
    split_csv: Path,
    tiles_npy: Path,
    stats_json: Path,
    batch_size: int,
    seed: int,
    device: torch.device,
    augmentation: str = "none",
) -> tuple[DataLoader, DataLoader, torch.Tensor]:
    """只增强训练集、只用训练标签计算权重；验证集保持原始分布。"""
    if augmentation not in AUGMENTATION_MODES:
        raise ValueError(f"未知增强方式：{augmentation}")
    # 翻转不会插值或裁掉 55×55 图中的细胞结构。
    geometry = [v2.RandomHorizontalFlip(), v2.RandomVerticalFlip()]
    if augmentation == "flips_photometric_geometric":
        geometry.extend([
            v2.RandomChoice([
                v2.Identity(),
                v2.RandomRotation((90, 90)),
                v2.RandomRotation((180, 180)),
                v2.RandomRotation((270, 270)),
            ]),
            # 灰度已按训练集统计量归一化；填充值 0 对应训练集均值。
            v2.RandomApply([
                v2.RandomAffine(
                    degrees=15, translate=(0.04, 0.04),
                    scale=(0.95, 1.05),
                    interpolation=v2.InterpolationMode.BILINEAR, fill=0.0,
                )
            ], p=0.5),
        ])
    train_transform = v2.Compose(geometry) if augmentation != "none" else None
    raw_transform = None
    if augmentation in {"flips_photometric", "flips_photometric_geometric"}:
        color_jitter = v2.ColorJitter(brightness=(0.8, 1.25), contrast=(0.9, 1.1))
        raw_transform = color_jitter
        if augmentation == "flips_photometric_geometric":
            # 原始浮点灰度约为 0.003；噪声须远小于像素尺度，且不裁剪到 [0, 1]。
            raw_transform = v2.Compose([
                color_jitter,
                v2.RandomApply([v2.GaussianBlur(3, sigma=(0.2, 0.6))], p=0.2),
                v2.RandomApply([v2.GaussianNoise(sigma=2e-5, clip=False)], p=0.2),
            ])
    common = (split_csv, tiles_npy, stats_json)
    train_set = LeukocyteDataset(*common, "train", transform=train_transform,
                                 raw_transform=raw_transform)
    # 验证批次带上原始 CSV 行号；Lightning 回调据此保存逐图块错分。
    val_set = LeukocyteDataset(*common, "val", include_index=True)
    counts = Counter(label for _index, label, _donor in train_set.samples)
    if len(counts) != len(CLASS_NAMES):
        raise ValueError("训练集缺少某个类别")

    # 平方根倒数权重缓和类别不均衡，避免直接倒数权重过于激进。
    class_counts = torch.tensor(
        [counts[i] for i in range(len(CLASS_NAMES))], dtype=torch.float32
    )
    weights = class_counts.rsqrt().to(device)
    weights /= weights.mean()
    print("训练集各类数量：", dict(zip(CLASS_NAMES, class_counts.int().tolist())))

    # Windows 上先用单进程读取，避免额外的数据加载进程带来文件映射问题。
    loader_options = {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **loader_options)
    val_loader = DataLoader(val_set, shuffle=False, **loader_options)
    return train_loader, val_loader, weights
