"""只用合成计划和预测验证亮度压力测试的边界与指标。"""

import copy
import tempfile
import unittest
from pathlib import Path

from leukocyte_analysis.evaluation.brightness import (
    _measure, _save_plots, _validate_plans, compare_brightness,
)


def plan(augmentation: str) -> dict:
    return {
        "method": "StratifiedGroupKFold", "source_manifest": "split.csv",
        "packed_tiles": "tiles.npy", "excluded_donors": ["CRF049", "CRF066"],
        "seed": 42, "training": {"epochs": 20, "augmentation": augmentation},
        "folds": [
            {"number": 1, "train_donors": ["B"], "val_donors": ["A"]},
            {"number": 2, "train_donors": ["A"], "val_donors": ["B"]},
        ],
    }


class TestBrightnessStress(unittest.TestCase):
    def test_only_matched_validation_folds_are_accepted(self) -> None:
        first, second = plan("flips"), plan("flips_photometric")
        _validate_plans(first, second)
        changed = copy.deepcopy(second)
        changed["folds"][0]["val_donors"] = ["CRF049"]
        with self.assertRaises(ValueError):
            _validate_plans(first, changed)
        changed = copy.deepcopy(second)
        changed["training"]["epochs"] = 12
        with self.assertRaisesRegex(ValueError, "超参数"):
            _validate_plans(first, changed)

    def test_metrics_and_plots_use_existing_prediction_format(self) -> None:
        predictions = [
            {"packed_index": 0, "donor": "A", "true_index": 0, "predicted_index": 0},
            {"packed_index": 1, "donor": "A", "true_index": 1, "predicted_index": 0},
            {"packed_index": 2, "donor": "B", "true_index": 2, "predicted_index": 2},
            {"packed_index": 3, "donor": "B", "true_index": 3, "predicted_index": 3},
        ]
        overall, classes, donors = _measure(predictions, "baseline", 1.0)
        self.assertEqual(overall["samples"], 4)
        self.assertAlmostEqual(overall["accuracy"], 0.75)
        self.assertEqual(classes[1]["recall"], 0)
        self.assertEqual({row["donor"] for row in donors}, {"A", "B"})
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            folder = Path(temporary)
            _save_plots([overall], classes, folder)
            self.assertGreater((folder / "overall.png").stat().st_size, 0)
            self.assertGreater((folder / "per_class_recall.png").stat().st_size, 0)

    def test_bad_factors_do_not_create_report(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            output = Path(temporary) / "report"
            with self.assertRaisesRegex(ValueError, "1.0"):
                compare_brightness(Path("first"), Path("second"), output,
                                   factors=(0.8, 1.2))
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
