"""验证训练入口挂载错分回调，并在验证轮次产出错误记录。"""

import csv
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from leukocyte_analysis.data.bbbc045_config import CLASS_NAMES
from leukocyte_analysis.training.error_logging import ValidationErrorLogger
from leukocyte_analysis.training.experiment import run_training
from _experiment_fixtures import four_sample_loader, temporary_run_root


class TestValidationErrorCallback(unittest.TestCase):
    """覆盖索引验证集触发回调及实际训练写出的 CSV。"""

    def test_indexed_validation_loader_gets_error_callback(self) -> None:
        """带原清单索引的验证集会自动挂载错分记录回调。"""
        train_loader = four_sample_loader()
        val_dataset = TensorDataset(torch.zeros(4, 1), torch.arange(4))
        val_dataset.include_index = True
        val_loader = DataLoader(val_dataset, batch_size=4)
        with temporary_run_root() as root:
            with patch("leukocyte_analysis.training.experiment.L.Trainer") as trainer:
                run_training(nn.Linear(1, 4), train_loader, val_loader,
                             torch.ones(4), torch.device("cpu"), 1,
                             root / "run", 0.01, 0.0)
        callbacks = trainer.call_args.kwargs["callbacks"]
        self.assertTrue(any(isinstance(item, ValidationErrorLogger) for item in callbacks))

    def test_indexed_validation_run_writes_error_records(self) -> None:
        """一轮真实训练后会生成逐轮错分和累计频率 CSV。"""
        train_loader = four_sample_loader()
        images = torch.zeros(4, 1)
        targets = torch.arange(4)
        val_dataset = TensorDataset(images, targets, torch.arange(4))
        val_dataset.include_index = True
        val_dataset.class_names = CLASS_NAMES
        raw_labels = ("B", "eosinophil", "monocyte", "neutrophil")
        val_dataset.source_by_index = {
            index: {"file": f"synthetic/{label}/batch1/ch5_1.tif",
                    "donor": "synthetic", "source_label": label,
                    "row": index, "col": 0}
            for index, label in enumerate(raw_labels)
        }
        with temporary_run_root() as root:
            output = root / "run"
            run_training(nn.Linear(1, 4), train_loader,
                         DataLoader(val_dataset, batch_size=4), torch.ones(4),
                         torch.device("cpu"), 1, output, 0.01, 0.0)
            error_dir = output / "validation_errors"
            self.assertTrue((error_dir / "epoch_000.csv").exists())
            self.assertTrue((error_dir / "frequency.csv").exists())
            with (error_dir / "epoch_000.csv").open(newline="", encoding="utf-8") as handle:
                errors = list(csv.DictReader(handle))
            self.assertGreater(len(errors), 0)
            self.assertTrue(all(row["split"] == "val" for row in errors))


if __name__ == "__main__":
    unittest.main()
