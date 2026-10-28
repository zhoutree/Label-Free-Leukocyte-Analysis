"""用合成特征验证 RMD 评分、状态恢复和已知误拒工作点，不读取正式图块。"""

import unittest
from importlib.util import find_spec

import numpy as np
import torch
from torch import nn

from leukocyte_analysis.evaluation.novelty import (
    ResNetFeatures, detector_state, fit_detector, known_threshold, outlier_scores,
    restore_detector,
)


class TestNovelty(unittest.TestCase):
    """检查第三方实现接入时的分数方向、序列化和类别边界。"""

    def test_working_point_preserves_ties_and_rejects_no_more_than_budget(self) -> None:
        """等于阈值自动保留；拒绝比例不因并列分数超过目标。"""
        scores = np.array([-8.0, -7.0, -7.0, -4.0, -1.0])
        threshold = known_threshold(scores, 0.3)
        self.assertLessEqual(np.mean(scores > threshold), 0.3)
        self.assertEqual(known_threshold(np.ones(4), 0.5), 1)
        for invalid in (0, 1, -0.1, float("nan")):
            with self.assertRaises(ValueError):
                known_threshold(scores, invalid)
        with self.assertRaises(ValueError):
            known_threshold(np.array([np.inf]), 0.1)

    @unittest.skipUnless(find_spec("pytorch_ood"), "需安装可选 review 依赖")
    def test_real_rmd_scores_and_tensor_state_round_trip(self) -> None:
        """与 RMD 定义核对方向；恢复状态后评分一致，不保存整个检测器对象。"""
        generator = torch.Generator().manual_seed(7)
        labels = torch.arange(4).repeat_interleave(12)
        features = torch.randn(48, 3, generator=generator).double() * 0.15
        features += labels[:, None].double() * 2
        detector = fit_detector(features, labels)
        queries = torch.tensor([[0.0, 0.0, 0.0], [30.0, 30.0, 30.0]], dtype=torch.float32)
        scores = outlier_scores(detector, queries)
        distances = []
        for center in detector.mu:
            residual = queries - center
            distances.append(((residual @ detector.precision) * residual).sum(1))
        residual = queries - detector.background_mu
        background = ((residual @ detector.background_precision) * residual).sum(1)
        expected = torch.stack(distances).min(0).values - background
        np.testing.assert_allclose(scores, expected.numpy())
        self.assertGreater(scores[1], scores[0])
        state = detector_state(detector)
        restored = restore_detector(state, 3)
        np.testing.assert_allclose(outlier_scores(restored, queries), scores)
        with self.assertRaises(ValueError):
            restore_detector(state, 5)
        with self.assertRaisesRegex(ValueError, "全部四类"):
            fit_detector(features[labels != 3], labels[labels != 3])

    def test_feature_adapter_preserves_original_head_and_logits(self) -> None:
        """一次前向提取描述，保持分类头共享和原始预测结果。"""
        backbone = nn.Module()
        backbone.add_module("avgpool", nn.AdaptiveAvgPool2d(1))
        backbone.add_module("fc", nn.Linear(1, 4))
        adapter = ResNetFeatures(backbone).eval()
        images = torch.randn(2, 1, 55, 55)
        expected = backbone.fc(backbone.avgpool(images).flatten(1))
        logits, features = adapter(images)
        torch.testing.assert_close(logits, expected)
        self.assertEqual(tuple(features.shape), (2, 1))
        self.assertIs(adapter.head.weight, backbone.fc.weight)


if __name__ == "__main__":
    unittest.main()
