"""验证清晰度代理指标的基本方向，不把它当作错标判定器。"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from leukocyte_analysis.data.audit import _inspect_tiles
from leukocyte_analysis.data.bbbc045_config import SOURCE_CLASS_NAMES
from leukocyte_analysis.data.quality import focus_proxy


class TestFocusProxy(unittest.TestCase):
    def test_blurred_edge_has_lower_score(self) -> None:
        sharp = np.zeros((55, 55), dtype=np.float32)
        sharp[14:41, 14:41] = 1
        blurred = gaussian_filter(sharp, sigma=2)
        self.assertGreater(focus_proxy(sharp), focus_proxy(blurred))

    def test_invalid_tile_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            focus_proxy(np.array([[np.nan]], dtype=np.float32))

    def test_audit_uses_training_threshold_without_removing_samples(self) -> None:
        sharp = np.zeros((55, 55), dtype=np.float32)
        sharp[14:41, 14:41] = 1
        blurred = gaussian_filter(sharp, sigma=2)
        records = [
            {"file": f"{label}.tif", "label": label, "split": "train",
             "row": "0", "col": "0"}
            for label in SOURCE_CLASS_NAMES
        ]
        records.append({"file": "blurred.tif", "label": SOURCE_CLASS_NAMES[0],
                        "split": "val", "row": "0", "col": "0"})
        tiles = np.stack([sharp] * len(SOURCE_CLASS_NAMES) + [blurred])
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            path = Path(temporary) / "tiles.npy"
            np.save(path, tiles)
            quality, _contrast, focus = _inspect_tiles(records, path)
        self.assertEqual(quality["nonfinite"], 0)
        self.assertEqual(focus["candidate_counts"]["val"][SOURCE_CLASS_NAMES[0]], 1)
        self.assertEqual(focus["auto_excluded"], 0)


if __name__ == "__main__":
    unittest.main()
