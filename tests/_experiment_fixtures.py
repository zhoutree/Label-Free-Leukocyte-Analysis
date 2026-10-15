"""为训练实验测试提供小规模合成批次和工作区内临时目录。"""

import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import DataLoader, TensorDataset


def four_sample_loader() -> DataLoader:
    """生成每类一个样本的最小批次，供回调配置测试使用。"""
    images = torch.zeros(4, 1)
    targets = torch.arange(4)
    return DataLoader(TensorDataset(images, targets), batch_size=4)


def balanced_four_class_loader() -> DataLoader:
    """生成两批均衡四分类样本，供真实短训练使用。"""
    images = torch.arange(8, dtype=torch.float32).reshape(-1, 1) / 8
    targets = torch.arange(8) % 4
    return DataLoader(TensorDataset(images, targets), batch_size=4)


@contextmanager
def temporary_run_root() -> Iterator[Path]:
    """在测试目录内创建自动清理的运行根目录。"""
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
        yield Path(temporary)
