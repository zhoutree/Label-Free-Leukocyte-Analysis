"""只用合成计划和预测验证亮度压力测试的边界与指标。"""

import copy
import tempfile
import unittest
from pathlib import Path

from leukocyte_analysis.data.bbbc045_config import CLASS_NAMES
from leukocyte_analysis.evaluation.brightness import (
    _measure, _validate_plans, compare_brightness,
)
from leukocyte_analysis.evaluation.plots import save_brightness_comparison


def plan(augmentation: str) -> dict:
    """构造两折且排除原测试受试者的最小亮度实验计划。"""
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
    """限制亮度比较只能在同一批折外样本上进行。"""

    def test_only_matched_validation_folds_are_accepted(self) -> None:
        """拒绝受试者分折或其他训练超参数不一致的两组运行。"""
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

    def test_relative_plan_paths_compare_resolved_sources(self) -> None:
        """相同相对字符串在不同目录下不一定指向同一份图块。"""
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            first, second = plan("flips"), plan("flips_photometric")
            with self.assertRaisesRegex(ValueError, "数据路径"):
                _validate_plans(first, second, run_dirs=(root / "a", root / "b"))

    def test_metrics_and_plots_use_existing_prediction_format(self) -> None:
        """总体、逐类和逐受试者统计可从同一预测格式生成。"""
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
            save_brightness_comparison([overall], classes, folder, CLASS_NAMES)
            self.assertGreater((folder / "overall.png").stat().st_size, 0)
            self.assertGreater((folder / "per_class_recall.png").stat().st_size, 0)

    def test_bad_factors_do_not_create_report(self) -> None:
        """缺少原亮度对照时应在创建报告目录前失败。"""
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            output = Path(temporary) / "report"
            with self.assertRaisesRegex(ValueError, "1.0"):
                compare_brightness(Path("first"), Path("second"), output,
                                   factors=(0.8, 1.2))
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
