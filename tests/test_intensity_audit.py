"""用合成 TIFF 图块验证亮度审计，不扫描完整 BBBC045。"""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from leukocyte_analysis.data.intensity_audit import audit_intensity


class TestIntensityAudit(unittest.TestCase):
    def test_batch_ratios_and_test_donor_exclusion(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            manifest = root / "split.csv"
            tiles_path = root / "tiles.npy"
            output = root / "report"
            records, tiles = [], []
            for batch, brightness in (("batch1", 0.003), ("batch2", 0.0036)):
                for _ in range(50):
                    records.append({"file": f"CRF022/T/{batch}/ch5_1.tif",
                                    "donor": "CRF022", "label": "T",
                                    "row": "0", "col": "0", "split": "train"})
                    tile = np.full((55, 55), brightness, dtype=np.float32)
                    tile[20:35, 20:35] += 0.001
                    tiles.append(tile)
            records.append({"file": "CRF049/monocyte/batch1/ch5_1.tif",
                            "donor": "CRF049", "label": "monocyte",
                            "row": "0", "col": "0", "split": "test"})
            tiles.append(np.full((55, 55), 0.01, dtype=np.float32))
            records.append({"file": "CRF066/monocyte/batch1/ch5_1.tif",
                            "donor": "CRF066", "label": "monocyte",
                            "row": "0", "col": "0", "split": "test"})
            tiles.append(np.full((55, 55), 0.0034, dtype=np.float32))
            records.append({"file": "CRF034/B/batch1/ch5_1.tif",
                            "donor": "CRF034", "label": "B",
                            "row": "0", "col": "0", "split": "val"})
            tiles.append(np.full((55, 55), 0.0034, dtype=np.float32))
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
            np.save(tiles_path, np.stack(tiles))

            settings = audit_intensity(manifest, tiles_path, output, chunk_size=13)
            self.assertEqual(settings["tiles_scanned"], 101)
            self.assertEqual(settings["split_counts"], {"train": 100, "val": 1})
            with (output / "within_donor_batch_pairs.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                pair = next(csv.DictReader(handle))
            self.assertAlmostEqual(float(pair["border_median_proxy_ratio_b_over_a"]),
                                   1.2, places=5)
            self.assertGreater((output / "largest_batch_shifts.png").stat().st_size, 0)
            with (output / "tile_metrics.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 101)
            self.assertEqual({row["donor"] for row in rows}, {"CRF022", "CRF034"})
            with self.assertRaises(FileExistsError):
                audit_intensity(manifest, tiles_path, output)

            all_output = root / "all_splits"
            all_settings = audit_intensity(manifest, tiles_path, all_output,
                                           chunk_size=13, include_test=True)
            self.assertEqual(all_settings["tiles_scanned"], 103)
            self.assertEqual(all_settings["split_counts"],
                             {"test": 2, "train": 100, "val": 1})
            self.assertEqual(all_settings["excluded_donors"], [])
            with (all_output / "tile_metrics.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                all_rows = list(csv.DictReader(handle))
            self.assertEqual({row["donor"] for row in all_rows if row["split"] == "test"},
                             {"CRF049", "CRF066"})
            self.assertGreater((all_output / "split_summary.csv").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
