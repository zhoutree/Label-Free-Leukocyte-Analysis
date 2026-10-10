"""只使用合成细胞验证独立评估，不读取保留的真实测试集。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from leukocyte_analysis.evaluation.runner import predict, save_report
from leukocyte_analysis.training.module import LeukocyteClassifier


class SyntheticCells:
    """模拟顺序数据集与清单索引，便于测试报告生成。"""

    def __init__(self) -> None:
        self.labels = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        self.tiles = np.stack([
            np.full((55, 55), 1 if index == 2 else label, dtype=np.float32)
            for index, label in enumerate(self.labels)
        ])
        self.samples = [(index, label, "synthetic")
                        for index, label in enumerate(self.labels)]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.from_numpy(self.tiles[index][None]), self.labels[index]

    def _get_tiles(self) -> np.ndarray:
        return self.tiles


class PixelClassifier(nn.Module):
    """用首个像素编码类别；第 2 个样本必定错分。"""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        labels = images[:, 0, 0, 0].long()
        return torch.nn.functional.one_hot(labels, num_classes=5).float() * 10


class TestEvaluation(unittest.TestCase):
    def test_report_and_error_images(self) -> None:
        dataset = SyntheticCells()
        model = LeukocyteClassifier(PixelClassifier(), torch.ones(5), 0.01, 0.0)
        predictions = predict(model, dataset, 4, torch.device("cpu"))
        self.assertEqual(len(predictions), 10)
        self.assertEqual(sum(row["true_index"] != row["predicted_index"]
                             for row in predictions), 1)

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            output = Path(temporary) / "test_evaluation"
            summary = save_report(dataset, predictions, output)
            self.assertAlmostEqual(summary["accuracy"], 0.9)
            for name in ("test_metrics.json", "test_classification_report.txt",
                         "test_per_class_metrics.csv",
                         "test_predictions.csv", "confusion_matrix_counts.png",
                         "confusion_matrix_row_normalized.png", "misclassified_examples.png"):
                self.assertGreater((output / name).stat().st_size, 0)
            with self.assertRaises(FileExistsError):
                output.mkdir(exist_ok=False)

    def test_failed_plot_does_not_publish_partial_report(self) -> None:
        dataset = SyntheticCells()
        model = LeukocyteClassifier(PixelClassifier(), torch.ones(5), 0.01, 0.0)
        predictions = predict(model, dataset, 4, torch.device("cpu"))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            output = Path(temporary) / "test_evaluation"
            with patch("leukocyte_analysis.evaluation.runner.save_error_examples",
                       side_effect=RuntimeError("plot failed")):
                with self.assertRaisesRegex(RuntimeError, "plot failed"):
                    save_report(dataset, predictions, output)
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(temporary).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
