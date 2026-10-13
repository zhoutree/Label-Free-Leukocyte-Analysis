"""只用合成清单检查受试者交叉验证，不启动完整训练。"""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from leukocyte_analysis.data.stats import write_train_stats
from leukocyte_analysis.data.bbbc045_config import TEST_DONORS, TRAIN_DONORS, VAL_DONORS
from leukocyte_analysis.data.cv_split import (
    build_donor_folds,
    write_fold_manifest,
)
from leukocyte_analysis.training.module import LeukocyteClassifier
from leukocyte_analysis.training.error_logging import ValidationErrorLogger
from leukocyte_analysis.training.cross_validation import (
    continue_cross_validation, run_cross_validation,
)
from leukocyte_analysis.evaluation.error_records import ERROR_FIELDS, write_csv


class TestDonorCrossValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.fields = ["file", "donor", "label", "row", "col", "split"]
        self.heldout = frozenset({"CRF049", "CRF066"})
        self.donors = [f"D{index}" for index in range(6)]
        self.records = [
            {"file": f"{donor}/{label}/batch1/ch5_1.tif", "donor": donor,
             "label": label, "row": "0", "col": "0",
             "split": "test" if donor in self.heldout else "train"}
            for donor in [*self.donors, "CRF049", "CRF066"]
            for label in ("B", "eosinophil", "monocyte", "neutrophil")
        ]

    def test_each_eligible_donor_is_validated_once(self) -> None:
        folds = build_donor_folds(self.records, 3, 42, self.heldout)
        self.assertEqual(len(folds), 3)
        self.assertEqual(set.union(*(set(fold.val_donors) for fold in folds)), set(self.donors))
        for fold in folds:
            self.assertFalse(fold.train_donors & fold.val_donors)
            self.assertFalse((fold.train_donors | fold.val_donors) & self.heldout)
            self.assertEqual(fold.train_donors | fold.val_donors, set(self.donors))

    def test_first_fold_can_be_completed_later_without_retraining(self) -> None:
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
            with fold_csv.open(newline="", encoding="utf-8") as handle:
                donors = {row["donor"] for row in csv.DictReader(handle)
                          if row["split"] == "val"}
            return object(), SimpleNamespace(dataset=SimpleNamespace(donors=donors)), torch.ones(4)

        def train(*args, **_kwargs):
            checkpoint = args[6] / "checkpoints" / "best.ckpt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()

        def predict(_model, dataset, _batch_size, _device):
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

    def test_crf101_years_stay_in_same_fold(self) -> None:
        records = [
            {"donor": donor, "label": label}
            for donor in (*self.donors, "CRF101_2014", "CRF101_2015")
            for label in ("B", "eosinophil", "monocyte", "neutrophil")
        ]
        folds = build_donor_folds(records, 3, 42)
        self.assertEqual(sum("CRF101_2014" in fold.val_donors for fold in folds), 1)
        for fold in folds:
            self.assertEqual("CRF101_2014" in fold.val_donors,
                             "CRF101_2015" in fold.val_donors)
            self.assertEqual("CRF101_2014" in fold.train_donors,
                             "CRF101_2015" in fold.train_donors)

    def test_fold_manifest_preserves_indices_and_excludes_crf049_monocyte(self) -> None:
        fold = build_donor_folds(self.records, 3, 42, self.heldout)[0]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            fold_csv = root / "split.csv"
            counts = write_fold_manifest(self.fields, self.records, fold, fold_csv)
            with fold_csv.open(newline="", encoding="utf-8") as handle:
                actual = list(csv.DictReader(handle))
            self.assertEqual(len(actual), len(self.records))
            self.assertEqual([row["file"] for row in actual],
                             [row["file"] for row in self.records])
            self.assertEqual(counts["excluded"], 8)
            self.assertTrue(all(row["split"] == "excluded" for row in actual
                                if row["donor"] in self.heldout))
            self.assertEqual(next(row["split"] for row in actual
                                  if row["donor"] == "CRF049" and row["label"] == "monocyte"),
                             "excluded")

            # 像素统计只能由该折 train 图块决定，不能借 val 或 excluded 的像素。
            tiles = np.stack([
                np.full((55, 55), index + 1, dtype=np.float32)
                for index in range(len(actual))
            ])
            tiles_npy = root / "tiles.npy"
            np.save(tiles_npy, tiles)
            stats = write_train_stats(fold_csv, tiles_npy, root / "stats.json")
            expected = tiles[[i for i, row in enumerate(actual) if row["split"] == "train"]]
            self.assertAlmostEqual(stats["mean"], float(expected.mean()), places=5)
            self.assertEqual(stats["train_tiles"], counts["train"])
            self.assertEqual(json.loads((root / "stats.json").read_text(encoding="utf-8"))["mean"],
                             stats["mean"])

    def test_indexed_batch_can_be_predicted(self) -> None:
        model = LeukocyteClassifier(nn.Flatten(start_dim=1), torch.ones(4), 1e-4, 0.0)
        images = torch.tensor([[[[0.0, 1.0, 0.0, 0.0]]]])
        targets = torch.tensor([1])
        result = model.predict_step((images, targets, torch.tensor([123])), 0)
        self.assertEqual(result[0].tolist(), [1])
        self.assertEqual(result[1].tolist(), [1])

    def test_validation_error_history_survives_resume(self) -> None:
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
