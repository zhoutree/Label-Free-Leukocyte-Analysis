"""验证错分追踪只关联正确的清单行，并统计重复出现的图块。"""

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from leukocyte_analysis.data.bbbc045_config import CLASS_NAMES
from leukocyte_analysis.evaluation.error_records import misclassified_rows
from leukocyte_analysis.training.error_logging import ValidationErrorLogger
from leukocyte_analysis.training.module import LeukocyteClassifier


class PixelClassifier(nn.Module):
    """读取首像素作为类别，构造确定性的合成错分。"""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        labels = images[:, 0, 0, 0].long()
        return torch.nn.functional.one_hot(labels, num_classes=4).float() * 10


class TestErrorLogging(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = SimpleNamespace(
            include_index=True,
            class_names=CLASS_NAMES,
            source_by_index={
                17: {"file": "CRF049/monocyte/batch1/ch5_1.tif",
                     "donor": "CRF049", "source_label": "monocyte",
                     "row": 5, "col": 8},
            },
        )

    def test_repeated_validation_error_is_counted(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            output = Path(temporary)
            callback = ValidationErrorLogger(self.dataset, output)
            trainer = SimpleNamespace(current_epoch=0, sanity_checking=False)
            for epoch, predicted in ((0, 0), (1, 1)):
                trainer.current_epoch = epoch
                callback.on_validation_epoch_start(trainer, None)
                callback.on_validation_batch_end(
                    trainer, None,
                    {"packed_index": [17], "true_index": [2],
                     "predicted_index": [predicted], "confidence": [0.95]},
                    None, 0,
                )
                callback.on_validation_epoch_end(trainer, None)

            with (output / "frequency.csv").open(newline="", encoding="utf-8") as handle:
                frequency = list(csv.DictReader(handle))
            self.assertEqual(len(frequency), 1)
            self.assertEqual(frequency[0]["wrong_epochs"], "2")
            self.assertEqual(frequency[0]["evaluated_epochs"], "2")
            self.assertEqual(frequency[0]["epochs"], "0;1")
            self.assertEqual(frequency[0]["predicted_labels"], "eosinophil:1;lymphocyte:1")
            with (output / "epoch_001.csv").open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["file"], "CRF049/monocyte/batch1/ch5_1.tif")
            self.assertEqual(row["row"], "5")
            self.assertEqual(row["col"], "8")

    def test_source_label_disagreement_is_rejected(self) -> None:
        prediction = {"packed_index": 17, "true_index": 1,
                      "predicted_index": 0, "confidence": 0.9}
        with self.assertRaisesRegex(ValueError, "原始标签"):
            misclassified_rows(self.dataset, [prediction], "val", 0)

    def test_validation_step_returns_only_wrong_indices(self) -> None:
        module = LeukocyteClassifier(PixelClassifier(), torch.ones(4), 0.01, 0.0)
        batch = (torch.zeros(2, 1, 1, 1), torch.tensor([0, 2]),
                 torch.tensor([16, 17]))
        with patch.object(module, "log"):
            output = module.validation_step(batch, 0)
        self.assertEqual(output["packed_index"], [17])
        self.assertEqual(output["true_index"], [2])
        self.assertEqual(output["predicted_index"], [0])
        self.assertGreater(output["confidence"][0], 0.9)


if __name__ == "__main__":
    unittest.main()
