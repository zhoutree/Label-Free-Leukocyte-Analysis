"""只用合成清单验证单折续训与三折汇总。"""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from leukocyte_analysis.data.bbbc045_config import TEST_DONORS, TRAIN_DONORS, VAL_DONORS
from leukocyte_analysis.data.cv_split import build_donor_folds, write_fold_manifest
from leukocyte_analysis.training.cross_validation import _metrics
from leukocyte_analysis.training.cv_resume import resume_cv_fold


class TestResumeFold(unittest.TestCase):
    def test_extends_only_second_fold_without_changing_source(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            source_dir = root / "original"
            source_dir.mkdir()
            donors = sorted(TRAIN_DONORS | VAL_DONORS | TEST_DONORS)
            records = [
                {"file": f"{donor}/B/batch1/ch5_1.tif", "donor": donor,
                 "label": "B", "row": "0", "col": "0",
                 "split": "test" if donor in TEST_DONORS else "train"}
                for donor in donors
            ]
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
            folds = build_donor_folds(records, 3, 42)
            second_donors = folds[1].val_donors
            wrong_donor = sorted(second_donors)[0]
            fold_results = []
            original_rows = {}
            for fold in folds:
                fold_dir = source_dir / f"fold_{fold.number:02d}"
                (fold_dir / "data").mkdir(parents=True)
                write_fold_manifest(list(records[0]), records, fold,
                                    fold_dir / "data" / "split.csv")
                (fold_dir / "data" / "train_stats.json").touch()
                checkpoints = fold_dir / "training" / "checkpoints"
                checkpoints.mkdir(parents=True)
                (checkpoints / "best.ckpt").touch()
                (checkpoints / "last.ckpt").touch()
                rows = [
                    {"packed_index": index, "donor": donor, "true_index": 0,
                     "predicted_index": 2 if donor == wrong_donor else 0}
                    for index, donor in enumerate(donors) if donor in fold.val_donors
                ]
                original_rows[fold.number] = rows
                with (fold_dir / "validation_predictions.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                fold_results.append({
                    "fold": fold.number, "val_donors": sorted(fold.val_donors),
                    "checkpoint": str(checkpoints / "best.ckpt"),
                    "metrics": _metrics(rows),
                })
            plan = {
                "source_manifest": str(manifest), "packed_tiles": str(root / "tiles.npy"),
                "excluded_donors": sorted(TEST_DONORS),
                "seed": 42,
                "training": {"epochs": 20, "batch_size": 128, "learning_rate": 1e-4,
                             "weight_decay": 0.01, "augmentation": "flips_photometric",
                             "early_stopping_patience": 5, "lr_scheduler": "plateau",
                             "label_smoothing": 0.05},
                "folds": [{"number": fold.number,
                           "train_donors": sorted(fold.train_donors),
                           "val_donors": sorted(fold.val_donors)} for fold in folds],
            }
            source_summary = source_dir / "summary.json"
            source_summary.write_text(json.dumps({
                "smoke_test": False, "plan": plan, "fold_results": fold_results,
            }), encoding="utf-8")
            original_content = source_summary.read_bytes()
            improved_rows = [{**row, "predicted_index": 0}
                             for row in original_rows[2]]

            def fake_train(*args, **kwargs):
                self.assertEqual(kwargs["resume_checkpoint"].name, "last.ckpt")
                self.assertEqual(args[5], 30)
                self.assertEqual(kwargs["schedule_epochs"], 20)
                self.assertEqual(kwargs["early_stopping_min_delta"], 0.0)
                self.assertEqual(kwargs["label_smoothing"], 0.05)
                checkpoint_dir = args[6] / "checkpoints"
                checkpoint_dir.mkdir(parents=True)
                (checkpoint_dir / "best.ckpt").touch()

            with (patch("leukocyte_analysis.training.cv_resume.make_loaders",
                        return_value=(object(), SimpleNamespace(dataset=object()),
                                      torch.ones(4))),
                  patch("leukocyte_analysis.training.cv_resume.build_resnet18"),
                  patch("leukocyte_analysis.training.cv_resume.run_training",
                        side_effect=fake_train) as training,
                  patch("leukocyte_analysis.training.cv_resume.load_classifier"),
                  patch("leukocyte_analysis.training.cv_resume.predict",
                        return_value=improved_rows)):
                output = root / "extended"
                summary = resume_cv_fold(source_dir, output, fold_number=2, epochs=30)

            training.assert_called_once()
            self.assertEqual(source_summary.read_bytes(), original_content)
            self.assertEqual(summary["continuation"]["selected"], "extended")
            self.assertEqual(summary["out_of_fold"]["samples"], len(TRAIN_DONORS | VAL_DONORS))
            self.assertEqual(summary["out_of_fold"]["accuracy"], 1.0)
            self.assertEqual(summary["fold_results"][0], fold_results[0])
            self.assertEqual(summary["fold_results"][2], fold_results[2])
            self.assertTrue((output / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
