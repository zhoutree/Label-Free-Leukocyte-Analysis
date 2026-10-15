"""验证训练产物中的数据划分、四分类检查点和指标曲线。"""

import csv
import json
import unittest
from unittest.mock import patch

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import nn

from leukocyte_analysis.data.bbbc045_config import CLASS_NAMES
from leukocyte_analysis.evaluation.runner import load_classifier
from leukocyte_analysis.training.experiment import run_training
from leukocyte_analysis.training.module import LeukocyteClassifier
from _experiment_fixtures import (
    balanced_four_class_loader,
    four_sample_loader,
    temporary_run_root,
)


class TestReporting(unittest.TestCase):
    """检查输出文件、分类头和 TensorBoard 指标可供后续分析。"""

    def test_regular_training_records_donor_grouping(self) -> None:
        """常规训练产物保存 CRF101 两年份归同组的划分记录。"""
        loader = four_sample_loader()
        record = {"group_overrides": {"CRF101_2014": "CRF101"},
                  "groups": [{"group": "CRF101", "split": "train",
                              "donors": ["CRF101_2014", "CRF101_2015"], "tiles": 2}]}
        with temporary_run_root() as root:
            output = root / "run"
            with patch("leukocyte_analysis.training.experiment.L.Trainer"):
                run_training(nn.Linear(1, 4), loader, loader, torch.ones(4),
                             torch.device("cpu"), 1, output, 1e-4, 0.0,
                             data_split_record=record)
            self.assertEqual(json.loads((output / "data_split.json").read_text(
                encoding="utf-8")), record)

    def test_four_class_module_uses_lymphocyte_metric(self) -> None:
        """分类器使用四类指标，B/T 合并后的首类是淋巴细胞。"""
        module = LeukocyteClassifier(nn.Linear(1, 4), torch.ones(4), 1e-4, 0.0)
        self.assertEqual(module.class_names, CLASS_NAMES)
        self.assertEqual(module.val_f1.num_classes, 4)

    def test_one_epoch_writes_four_class_checkpoint_and_curves(self) -> None:
        """真实短训练同时留下可恢复权重、CSV 和 TensorBoard 曲线。"""
        loader = balanced_four_class_loader()
        images = loader.dataset.tensors[0]
        model = nn.Linear(1, 4)
        weights = torch.tensor([1.0, 2.0, 1.0, 1.0])

        with temporary_run_root() as root:
            output = root / "run"
            run_training(model, loader, loader, weights, torch.device("cpu"),
                         1, output, 0.01, 0.0, lr_scheduler="cosine")

            with (output / "csv" / "metrics.csv").open(newline="", encoding="utf-8") as history:
                rows = list(csv.DictReader(history))
            self.assertTrue(any(row.get("train_weighted_loss") for row in rows))
            self.assertTrue(any(row.get("val_macro_f1") for row in rows))
            self.assertTrue(any(row.get("lr-AdamW") for row in rows))
            self.assertTrue((output / "checkpoints" / "best.ckpt").exists())
            self.assertTrue((output / "checkpoints" / "last.ckpt").exists())

            restored = load_classifier(output / "checkpoints" / "best.ckpt", nn.Linear(1, 4))
            self.assertEqual(restored.class_names, CLASS_NAMES)
            self.assertEqual(tuple(restored(images).shape), (8, 4))

            events = EventAccumulator(str(output / "tensorboard"))
            events.Reload()
            for name in ("train_loss", "val_loss", "train_weighted_loss",
                         "val_macro_f1", "val_recall_lymphocyte"):
                self.assertEqual(len(events.Scalars(name)), 1)


if __name__ == "__main__":
    unittest.main()
