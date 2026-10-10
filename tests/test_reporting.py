"""用少量合成样本验证 Lightning 的训练日志和检查点。"""

import csv
import tempfile
import unittest
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from leukocyte_analysis.training.experiment import run_training
from leukocyte_analysis.evaluation.runner import load_classifier


class TestReporting(unittest.TestCase):
    def test_one_epoch_writes_comparable_losses_and_curves(self) -> None:
        # 五类各两个样本，避免借用正式数据或启动完整 ResNet 训练。
        images = torch.arange(10, dtype=torch.float32).reshape(-1, 1) / 10
        targets = torch.arange(10) % 5
        loader = DataLoader(TensorDataset(images, targets), batch_size=5)
        model = nn.Linear(1, 5)
        weights = torch.tensor([1.0, 2.0, 1.0, 1.0, 1.0])

        # 使用项目内临时目录，兼容只允许写入工作区的运行环境。
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            output = Path(temporary) / "run"
            run_training(model, loader, loader, weights, torch.device("cpu"),
                         1, output, 0.01, 0.0)

            with (output / "csv" / "metrics.csv").open(newline="", encoding="utf-8") as history:
                rows = list(csv.DictReader(history))
            self.assertTrue(any(row.get("train_weighted_loss") for row in rows))
            self.assertTrue(any(row.get("val_macro_f1") for row in rows))
            self.assertTrue((output / "checkpoints" / "best.ckpt").exists())
            self.assertTrue((output / "checkpoints" / "last.ckpt").exists())

            # 被忽略的模型结构在加载时显式提供；确认检查点能独立恢复。
            restored = load_classifier(output / "checkpoints" / "best.ckpt", nn.Linear(1, 5))
            self.assertEqual(tuple(restored(images).shape), (10, 5))

            events = EventAccumulator(str(output / "tensorboard"))
            events.Reload()
            for name in ("train_loss", "val_loss", "train_weighted_loss",
                         "val_macro_f1", "val_recall_B"):
                self.assertEqual(len(events.Scalars(name)), 1)


if __name__ == "__main__":
    unittest.main()
