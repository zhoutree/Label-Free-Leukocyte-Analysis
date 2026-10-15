"""验证清晰度代理指标的基本方向，不把它当作错标判定器。"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from leukocyte_analysis.data.audit import _inspect_tiles, _pixel_issue
from leukocyte_analysis.data.bbbc045_config import SOURCE_CLASS_NAMES
from leukocyte_analysis.data.quality import focus_proxy


class TestFocusProxy(unittest.TestCase):
    """区分异常像素和清晰度提示，均不自动改动样本。"""

    def test_pixel_issues_are_counted_separately_from_focus(self) -> None:
        """非有限、全零和常数图块分别计数，不混入模糊度。"""
        self.assertEqual(_pixel_issue(np.full((55, 55), np.nan)), "nonfinite")
        self.assertEqual(_pixel_issue(np.zeros((55, 55))), "all_zero")
        self.assertEqual(_pixel_issue(np.ones((55, 55))), "constant")
        self.assertIsNone(_pixel_issue(np.eye(55)))

    def test_blurred_edge_has_lower_score(self) -> None:
        """同一形状的高斯模糊版本应有更低清晰度代理值。"""
        sharp = np.zeros((55, 55), dtype=np.float32)
        sharp[14:41, 14:41] = 1
        blurred = gaussian_filter(sharp, sigma=2)
        self.assertGreater(focus_proxy(sharp), focus_proxy(blurred))

    def test_invalid_tile_is_rejected(self) -> None:
        """清晰度计算拒绝非有限像素。"""
        with self.assertRaises(ValueError):
            focus_proxy(np.array([[np.nan]], dtype=np.float32))

    def test_audit_uses_training_threshold_without_removing_samples(self) -> None:
        """模糊阈值只由训练图块决定，候选图块不会被自动删除。"""
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
