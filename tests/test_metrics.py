"""确认共享指标保留类别顺序及宏平均的计算口径。"""

import unittest

from leukocyte_analysis.evaluation.metrics import classification_metrics


class TestClassificationMetrics(unittest.TestCase):
    def test_all_classes_remain_in_report_when_one_is_missing(self) -> None:
        result = classification_metrics([0, 0, 1], [0, 1, 1], ("A", "B", "C"))
        self.assertEqual(result["samples"], 3)
        self.assertEqual(result["class_names"], ["A", "B", "C"])
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["macro_f1"], 4 / 9)
        self.assertEqual(result["classification_report"]["C"]["support"], 0)

    def test_mismatched_lengths_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            classification_metrics([0, 1], [0], ("A", "B"))
