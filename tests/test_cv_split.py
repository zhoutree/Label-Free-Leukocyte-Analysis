"""验证受试者分折、留出集合及折清单的训练统计边界。"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from leukocyte_analysis.data.cv_split import build_donor_folds, write_fold_manifest
from leukocyte_analysis.data.stats import write_train_stats


class TestDonorFoldSplit(unittest.TestCase):
    """用合成清单约束每位受试者的验证次数和折内数据归属。"""

    def setUp(self) -> None:
        """准备含六个可分折受试者和两个保留测试受试者的清单。"""
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
        """每位可用受试者只在一折验证，保留测试受试者从不进入折内。"""
        folds = build_donor_folds(self.records, 3, 42, self.heldout)
        self.assertEqual(len(folds), 3)
        self.assertEqual(set.union(*(set(fold.val_donors) for fold in folds)), set(self.donors))
        for fold in folds:
            self.assertFalse(fold.train_donors & fold.val_donors)
            self.assertFalse((fold.train_donors | fold.val_donors) & self.heldout)
            self.assertEqual(fold.train_donors | fold.val_donors, set(self.donors))

    def test_crf101_years_stay_in_same_fold(self) -> None:
        """CRF101 的两个年份作为同一受试者组，不跨训练／验证。"""
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

    def test_fold_manifest_preserves_indices_and_excludes_heldout_donors(self) -> None:
        """折清单保持原索引，统计量仅由该折训练图块计算。"""
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


if __name__ == "__main__":
    unittest.main()
