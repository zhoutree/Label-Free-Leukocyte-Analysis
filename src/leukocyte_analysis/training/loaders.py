"""构建 BBBC045 训练/验证数据加载器和类别权重。"""

from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from ..data.bbbc045_config import CLASS_NAMES
from ..data.dataset import LeukocyteDataset


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
    if augmentation not in {"none", "flips"}:
        raise ValueError(f"未知增强方式：{augmentation}")
    # 翻转不会插值或裁掉 55×55 图中的细胞结构。
    train_transform = (
        v2.Compose([v2.RandomHorizontalFlip(), v2.RandomVerticalFlip()])
        if augmentation == "flips" else None
    )
    common = (split_csv, tiles_npy, stats_json)
    train_set = LeukocyteDataset(*common, "train", transform=train_transform)
    val_set = LeukocyteDataset(*common, "val")
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
