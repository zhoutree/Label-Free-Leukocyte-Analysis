"""检验图片复核规则与小型端到端报告；不启动真实数据推理或训练。"""

import csv
import json
import sys
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from leukocyte_analysis.evaluation.cell_review import ReviewPolicy
from leukocyte_analysis.data.split import read_reference_manifest
from leukocyte_analysis.evaluation.quality_reference import calibrate_quality
from leukocyte_analysis.evaluation.novelty import ResNetFeatures
from leukocyte_analysis.evaluation.review_runner import (
    fit_reference, load_reference, review_tiles,
)
from leukocyte_analysis.review_cells import parse_args


class TestCellReview(unittest.TestCase):
    """真实标签不参与决策，质量判废与 RMD 特征待定分开。"""

    def test_decisions_are_independent_and_boundaries_are_explicit(self) -> None:
        """RMD 等于上限保留，超过上限待定；严重模糊优先输出不可用。"""
        policy = ReviewPolicy(-2, focus_min=0.2, focus_review_min=0.5)
        quality = {"normalized_tenengrad": 0.5, "contrast_std": 1.0}
        self.assertEqual(policy.assess(2, -2, quality)["decision"], "accepted")
        novel = policy.assess(2, -1, quality)
        self.assertEqual(novel["review_reason"], "feature_outlier")
        self.assertEqual(novel["decision_label"], "uncertain")
        quality["normalized_tenengrad"] = 0.1
        bad = policy.assess(2, -1, quality)
        self.assertEqual(bad["decision"], "unusable")
        self.assertIn("severe_blur_candidate", bad["review_reason"])
        self.assertIn("feature_outlier", bad["review_reason"])
        quality["normalized_tenengrad"] = 0.3
        self.assertEqual(policy.assess(2, -3, quality)["quality_status"], "needs_review")

    def test_no_blur_threshold_means_unassessed_not_quality_pass(self) -> None:
        """没有人工核定阈值时只记录评分，常数图的无结构原因仍独立记录。"""
        policy = ReviewPolicy(1)
        quality = {"normalized_tenengrad": 0.01, "contrast_std": 0.1}
        self.assertEqual(policy.assess(0, 0, quality)["quality_status"], "not_assessed")
        quality["contrast_std"] = 0
        self.assertEqual(policy.assess(0, 0, quality)["review_reason"], "no_contrast")
        for kwargs in ({"focus_min": -1}, {"focus_min": 2, "focus_review_min": 1},
                       {"focus_review_min": float("nan")}, {"ood_threshold": float("inf")}):
            with self.assertRaises(ValueError):
                ReviewPolicy(**{"ood_threshold": 0, **kwargs})

    def test_cli_rejects_invalid_rules_before_loading_data(self) -> None:
        """命令入口在加载模型前拒绝无效批量大小和矛盾的质量参数。"""
        base = ["review", "predict", "--checkpoint", "a", "--reference", "b",
                "--output-dir", "out", "--tiles-npy", "cells.npy"]
        for extra in (["--batch-size", "0"], ["--focus-min", "nan"], ["--confidence-min", "0.8"],
                      ["--focus-min", "2", "--quality-reference", "quality.json"],
                      ["--focus-min", "2", "--focus-review-min", "1"]):
            with patch.object(sys, "argv", base + extra), patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    parse_args()
        with patch.object(sys, "argv", ["review", "calibrate-quality", "--validation-scores",
                                         "scores.csv", "--output-dir", "out"]):
            self.assertEqual(parse_args().quality_reject_rate, 0.01)

    @unittest.skipUnless(find_spec("pytorch_ood"), "需安装可选 review 依赖")
    def test_synthetic_fit_and_unlabeled_application(self) -> None:
        """真实第三方 RMD 加合成网络完成闭环，输出不覆盖输入或旧报告。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, tiles_file, stats = root / "split.csv", root / "tiles.npy", root / "stats.json"
            checkpoint = root / "model.ckpt"
            checkpoint.write_bytes(b"synthetic-checkpoint")
            records, tiles = [], []
            for split, repeats in (("train", 3), ("val", 1)):
                for label, name in enumerate(("B", "eosinophil", "monocyte", "neutrophil")):
                    for repeat in range(repeats):
                        records.append({"file": f"{len(tiles)}.tif", "donor": split,
                                        "label": name, "row": "0", "col": "0", "split": split})
                        image = np.full((55, 55), label + repeat * 0.01, dtype=np.float32)
                        image[15:40, 15:40] += 0.5
                        tiles.append(image)
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
            np.save(tiles_file, np.stack(tiles))
            stats.write_text(json.dumps({"mean": 0, "std": 1}), encoding="utf-8")
            backbone = nn.Module()
            backbone.add_module("avgpool", nn.AdaptiveAvgPool2d(1))
            backbone.add_module("fc", nn.Linear(1, 4))
            network = ResNetFeatures(backbone).eval()
            with patch("leukocyte_analysis.evaluation.review_runner._load_network",
                       side_effect=lambda _path, device: network.to(device)):
                summary = fit_reference(checkpoint, manifest, tiles_file, stats, root / "reference", 0.1, 3)
                self.assertEqual(summary["train_samples"], 12)
                self.assertEqual(summary["calibration_samples"], 4)
                self.assertLessEqual(summary["known_reject_rate_observed"], 0.1)
                artifact = root / "reference/rmd.pt"
                unknown_input = root / "unlabeled.npy"
                np.save(unknown_input, np.stack([tiles[0], np.zeros((55, 55), dtype=np.float32)]))
                result = review_tiles(checkpoint, artifact, unknown_input, root / "review", batch_size=1)
                self.assertEqual(result["samples"], 2)
                self.assertEqual(result["decisions"]["unusable"], 1)
                with (root / "review/predictions_with_review.csv").open(encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertNotIn("true_index", rows[0])
                self.assertEqual(rows[1]["review_reason"].split(";")[0], "no_contrast")
                quality = calibrate_quality(root / "reference/validation_scores.csv",
                                            root / "quality", split_csv=manifest)
                calibrated = review_tiles(checkpoint, artifact, unknown_input,
                                          root / "calibrated", quality_reference=
                                          root / "quality/quality_threshold.json")
                self.assertEqual(calibrated["policy"]["focus_min"], quality["focus_min"])
                self.assertEqual(calibrated["quality_thresholds"],
                                 "validation_tail_calibrated_not_quality_validated")
                with self.assertRaisesRegex(ValueError, "不能同时"):
                    review_tiles(checkpoint, artifact, unknown_input, root / "conflict",
                                 focus_min=1, quality_reference=root / "quality/quality_threshold.json")
                with self.assertRaises(FileExistsError):
                    review_tiles(checkpoint, artifact, unknown_input, root / "review")
                np.save(unknown_input, np.full((1, 55, 55), np.nan, dtype=np.float32))
                with self.assertRaises(ValueError):
                    review_tiles(checkpoint, artifact, unknown_input, root / "failed")
                self.assertFalse((root / "failed").exists())
            checkpoint.write_bytes(b"different-checkpoint")
            with self.assertRaisesRegex(ValueError, "不匹配"):
                load_reference(artifact, checkpoint)
            records[-1]["donor"] = "CRF049"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
            with self.assertRaisesRegex(ValueError, "测试受试者"):
                read_reference_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
