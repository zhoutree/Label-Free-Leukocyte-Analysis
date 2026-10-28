"""验证质量校准来源隔离、类内预算和参考兼容性，不加载模型。"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from leukocyte_analysis.data.integrity import file_sha256
from leukocyte_analysis.evaluation.quality_reference import (
    calibrate_focus, calibrate_quality, load_quality_threshold,
)


def _write_table(path: Path, rows: list[dict]) -> None:
    """在临时目录按首行字段构造小型清单或评分表。"""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _inputs(root: Path) -> tuple[Path, Path, list[dict], list[dict]]:
    """构造各含四类的验证、训练、测试记录，CSV 行号作为图块索引。"""
    manifest, scores = root / "split.csv", root / "scores.csv"
    names = ("B", "eosinophil", "monocyte", "neutrophil")
    records = [{"file": f"{donor}/{name}/batch1/ch5_1.tif", "donor": donor,
                "label": name, "row": "0", "col": "0", "split": split}
               for donor, split in (("CRF034", "val"), ("CRF022", "train"), ("CRF049", "test"))
               for name in names]
    rows = [{"packed_index": index, "donor": "CRF034", "true_index": index,
             "normalized_tenengrad": index + 4} for index in range(4)]
    _write_table(manifest, records)
    _write_table(scores, rows)
    return manifest, scores, records, rows


class TestQualityReference(unittest.TestCase):
    """阈值仅校准完整验证评分，不因手工命名 CSV 而跳过来源核对。"""

    def test_focus_calibration_limits_each_class_and_retains_ties(self) -> None:
        """不同类分布、并列及不足百张时都不超出各类预算。"""
        labels = np.repeat(np.arange(4), 100)
        scores = np.tile(np.arange(100), 4) + labels * 10
        result = calibrate_focus(scores, labels)
        self.assertEqual(result["focus_min"], 1)
        self.assertEqual(result["rejected"], 1)
        self.assertTrue(all(row["reject_rate_observed"] <= 0.01 for row in result["per_class"]))
        self.assertEqual(calibrate_focus([2] * 4, range(4))["rejected"], 0)
        for values, targets, rate in (([], [], 0.01), ([1, 2], [0, 1], 0.01),
                                      ([1, 2, 3, np.nan], range(4), 0.01),
                                      ([1, 2, 3, 4], range(4), 1)):
            with self.assertRaises(ValueError):
                calibrate_focus(values, targets, rate)

    def test_quality_artifact_round_trip_and_legacy_loading(self) -> None:
        """来源摘要可追溯、旧格式仍可读，校准不覆盖输入或既有目录。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, scores, _, _ = _inputs(root)
            original = scores.read_bytes()
            result = calibrate_quality(scores, root / "quality", split_csv=manifest)
            artifact = root / "quality/quality_threshold.json"
            self.assertEqual(load_quality_threshold(artifact), result["focus_min"])
            self.assertEqual(result["source_manifest"]["sha256"], file_sha256(manifest))
            self.assertEqual(scores.read_bytes(), original)
            with self.assertRaises(FileExistsError):
                calibrate_quality(scores, root / "quality", split_csv=manifest)
            del result["source_manifest"]
            artifact.write_text(json.dumps(result), encoding="utf-8")
            self.assertEqual(load_quality_threshold(artifact), result["focus_min"])
            result["method"] = "unknown"
            artifact.write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "不匹配"):
                load_quality_threshold(artifact)

    def test_non_validation_duplicate_missing_and_mismatched_rows_are_rejected(self) -> None:
        """训练／测试索引、重复缺失、错受试者、错标签或非 val 列均在发布前失败。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, scores, _, rows = _inputs(root)
            variants = [rows[:3], rows[:3] + [rows[0]],
                        [{**row, "packed_index": row["packed_index"] + 4,
                          "donor": "CRF022"} for row in rows],
                        [{**row, "packed_index": row["packed_index"] + 8,
                          "donor": "CRF049"} for row in rows],
                        [{**row, "donor": "CRF049"} for row in rows],
                        [{**row, "true_index": (row["true_index"] + 1) % 4} for row in rows],
                        [{**row, "split": "train"} for row in rows],
                        [{"true_index": row["true_index"],
                          "normalized_tenengrad": row["normalized_tenengrad"]} for row in rows]]
            for number, variant in enumerate(variants):
                with self.subTest(number=number):
                    _write_table(scores, variant)
                    output = root / f"invalid_{number}"
                    with self.assertRaises(ValueError):
                        calibrate_quality(scores, output, split_csv=manifest)
                    self.assertFalse(output.exists())

    def test_heldout_donor_and_cross_year_leakage_in_manifest_are_rejected(self) -> None:
        """源清单也必须排除测试受试者，并保持 CRF101 两个年份同组。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, scores, records, _ = _inputs(root)
            heldout = [{**row, "donor": "CRF049"} if row["split"] == "val" else row
                       for row in records]
            years = [{**row, "donor": "CRF101_2014" if row["split"] == "val" else
                      "CRF101_2015"} if row["split"] in {"val", "train"} else row
                     for row in records]
            for number, invalid in enumerate((heldout, years)):
                _write_table(manifest, invalid)
                with self.assertRaises(ValueError):
                    calibrate_quality(scores, root / f"bad_manifest_{number}", split_csv=manifest)
