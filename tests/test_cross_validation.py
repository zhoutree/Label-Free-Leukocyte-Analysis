"""用合成清单验证交叉验证可先运行一折，再继续完成剩余折。"""

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from leukocyte_analysis.data.bbbc045_config import TEST_DONORS, TRAIN_DONORS, VAL_DONORS
from leukocyte_analysis.training.cross_validation import (
    continue_cross_validation, run_cross_validation,
)


class TestCrossValidationOrchestration(unittest.TestCase):
    """检查初次运行、续跑和汇总之间的训练调用与输出。"""

    def test_first_fold_can_be_completed_later_without_retraining(self) -> None:
        """先跑一折再继续时，不重训首折且折外预测覆盖所有可用受试者。"""
        fields = ["file", "donor", "label", "row", "col", "split"]
        labels = ("B", "eosinophil", "monocyte", "neutrophil")
        records = [
            {"file": f"{donor}/{label}/batch1/ch5_1.tif", "donor": donor,
             "label": label, "row": "0", "col": "0",
             "split": "test" if donor in TEST_DONORS else "train"}
            for donor in sorted(TRAIN_DONORS | VAL_DONORS | TEST_DONORS)
            for label in labels
        ]

        def loaders(fold_csv, *_args):
            """从折清单提取验证受试者，替代实际图块加载。"""
            with fold_csv.open(newline="", encoding="utf-8") as handle:
                donors = {row["donor"] for row in csv.DictReader(handle)
                          if row["split"] == "val"}
            return object(), SimpleNamespace(dataset=SimpleNamespace(donors=donors)), torch.ones(4)

        def train(*args, **_kwargs):
            """只生成本折检查点标记，不启动真实模型训练。"""
            checkpoint = args[6] / "checkpoints" / "best.ckpt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()

        def predict(_model, dataset, _batch_size, _device):
            """按折验证受试者返回可追溯且全部正确的合成预测。"""
            return [
                {"packed_index": index, "donor": record["donor"],
                 "true_index": labels.index(record["label"]),
                 "predicted_index": labels.index(record["label"])}
                for index, record in enumerate(records)
                if record["donor"] in dataset.donors
            ]

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            tiles = root / "tiles.npy"
            tiles.touch()
            output = root / "run"
            module = "leukocyte_analysis.training.cross_validation"
            with (patch(f"{module}.read_manifest", return_value=(fields, records)),
                  patch(f"{module}.write_train_stats"),
                  patch(f"{module}.make_loaders", side_effect=loaders),
                  patch(f"{module}.build_resnet18", return_value=nn.Identity()),
                  patch(f"{module}.run_training", side_effect=train) as training,
                  patch(f"{module}.load_classifier", return_value=nn.Identity()),
                  patch(f"{module}.predict", side_effect=predict),
                  patch(f"{module}.L.seed_everything")):
                first = run_cross_validation(
                    root / "split.csv", tiles, output, n_splits=3, seed=42,
                    epochs=30, batch_size=128, learning_rate=1e-4,
                    weight_decay=0.01, initialization="random",
                    augmentation="flips_photometric_geometric",
                    early_stopping_patience=15, lr_scheduler="plateau",
                    early_stopping_min_delta=0.0, label_smoothing=0.05,
                    first_fold_only=True,
                )
                self.assertTrue(first["partial"])
                self.assertEqual(first["plan"]["training"]["label_smoothing"], 0.05)
                self.assertEqual(training.call_args_list[0].kwargs["label_smoothing"], 0.05)
                self.assertEqual(training.call_args_list[0].kwargs["early_stopping_min_delta"], 0.0)
                self.assertEqual(first["completed_folds"], 1)
                self.assertEqual(training.call_count, 1)
                self.assertFalse((output / "fold_02").exists())

                complete = continue_cross_validation(output)
                self.assertEqual(training.call_count, 3)
                self.assertTrue(all(call.kwargs["label_smoothing"] == 0.05
                                    for call in training.call_args_list))
                self.assertFalse(complete.get("partial", False))
                self.assertEqual(len(complete["fold_results"]), 3)
                self.assertEqual(complete["out_of_fold"]["samples"],
                                 4 * len(TRAIN_DONORS | VAL_DONORS))
                with self.assertRaisesRegex(ValueError, "尚未完成"):
                    continue_cross_validation(output)


if __name__ == "__main__":
    unittest.main()
