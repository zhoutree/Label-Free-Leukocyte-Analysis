"""验证错分记录正确关联清单、跨轮累计，并在续训后延续历史。"""

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from leukocyte_analysis.data.bbbc045_config import CLASS_NAMES
from leukocyte_analysis.evaluation.error_records import ERROR_FIELDS, misclassified_rows
from leukocyte_analysis.reporting import write_csv
from leukocyte_analysis.training.error_logging import ValidationErrorLogger
from leukocyte_analysis.training.module import LeukocyteClassifier


class PixelClassifier(nn.Module):
    """读取首像素作为类别，构造确定性的合成错分。"""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """把每张图的首像素编码成四类 logits。"""
        labels = images[:, 0, 0, 0].long()
        return torch.nn.functional.one_hot(labels, num_classes=4).float() * 10


class TestErrorLogging(unittest.TestCase):
    """核对验证错分的原图定位、轮次累计与续训历史。"""

    def setUp(self) -> None:
        """准备带原 TIFF 路径和网格坐标的单张合成图块。"""
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
        """同一图块连续两轮错分时频率表应记录两个轮号。"""
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
        """预测记录与清单原始标签不一致时拒绝生成错分表。"""
        prediction = {"packed_index": 17, "true_index": 1,
                      "predicted_index": 0, "confidence": 0.9}
        with self.assertRaisesRegex(ValueError, "原始标签"):
            misclassified_rows(self.dataset, [prediction], "val", 0)

    def test_correct_prediction_still_checks_source_label(self) -> None:
        """过期 CSV 即使声称预测正确，也不能跳过真实标签核对。"""
        prediction = {"packed_index": 17, "true_index": 0,
                      "predicted_index": 0, "confidence": 0.9}
        with self.assertRaisesRegex(ValueError, "原始标签"):
            misclassified_rows(self.dataset, [prediction], "val", 0)

    def test_validation_step_returns_only_wrong_indices(self) -> None:
        """模型验证步骤只把错分索引及置信度交给记录回调。"""
        module = LeukocyteClassifier(PixelClassifier(), torch.ones(4), 0.01, 0.0)
        batch = (torch.zeros(2, 1, 1, 1), torch.tensor([0, 2]),
                 torch.tensor([16, 17]))
        with patch.object(module, "log"):
            output = module.validation_step(batch, 0)
        self.assertEqual(output["packed_index"], [17])
        self.assertEqual(output["true_index"], [2])
        self.assertEqual(output["predicted_index"], [0])
        self.assertGreater(output["confidence"][0], 0.9)

    def test_validation_error_history_survives_resume(self) -> None:
        """新轮次追加错分后仍保留续训前的累计历史。"""
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            error_dir = Path(temporary) / "validation_errors"
            write_csv(error_dir / "epoch_000.csv", [{
                "split": "val", "epoch": 0, "packed_index": 12,
                "file": "D1/B/batch1/ch5_1.tif", "donor": "D1", "row": 0,
                "col": 0, "source_label": "B", "true_label": "lymphocyte",
                "predicted_label": "monocyte", "confidence": 0.9,
            }], ERROR_FIELDS)
            dataset = type("DatasetStub", (), {"include_index": True})()
            callback = ValidationErrorLogger(dataset, error_dir)
            self.assertEqual(callback._completed_epochs, 1)
            self.assertEqual(callback._history[12]["epochs"], [0])
            self.assertEqual(callback._history[12]["predictions"]["monocyte"], 1)
            self.assertIsInstance(callback._history[12]["source"]["packed_index"], int)
            new_error = {
                "split": "val", "epoch": 1, "packed_index": 13,
                "file": "D1/B/batch1/ch5_1.tif", "donor": "D1", "row": 0,
                "col": 1, "source_label": "B", "true_label": "lymphocyte",
                "predicted_label": "monocyte", "confidence": 0.8,
            }
            trainer = SimpleNamespace(sanity_checking=False, current_epoch=1)
            with patch("leukocyte_analysis.training.error_logging.misclassified_rows",
                       return_value=[new_error]):
                callback.on_validation_epoch_end(trainer, None)
            with (error_dir / "frequency.csv").open(newline="", encoding="utf-8") as handle:
                frequency = list(csv.DictReader(handle))
            self.assertEqual({int(row["packed_index"]) for row in frequency}, {12, 13})
            self.assertTrue(all(row["evaluated_epochs"] == "2" for row in frequency))


if __name__ == "__main__":
    unittest.main()
