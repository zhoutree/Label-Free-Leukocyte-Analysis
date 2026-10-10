"""训练入口和指标的轻量测试，不加载完整图像数据。"""

import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torchmetrics.classification import MulticlassF1Score, MulticlassRecall
from torchvision.models import ResNet18_Weights, resnet18

from leukocyte_analysis.models.resnet import build_resnet18
from leukocyte_analysis.train import PROJECT_ROOT, parse_args


class TestTrainingConfiguration(unittest.TestCase):
    def test_default_data_paths(self) -> None:
        with patch.object(sys, "argv", ["leukocyte-train"]):
            args = parse_args()
        self.assertEqual(args.split_csv, PROJECT_ROOT / "data/processed/bbbc045_split.csv")
        self.assertEqual(args.stats_json, PROJECT_ROOT / "data/processed/bbbc045_train_stats.json")
        self.assertEqual(args.init, "imagenet")
        self.assertEqual(args.lr, 1e-4)
        self.assertEqual(args.weight_decay, 0.01)

    def test_random_initialization_can_be_selected(self) -> None:
        with patch.object(sys, "argv", ["leukocyte-train", "--init", "random", "--weight-decay", "0"]):
            args = parse_args()
        self.assertEqual(args.init, "random")
        self.assertEqual(args.weight_decay, 0.0)

    def test_pretrained_backbone_and_small_image_layers(self) -> None:
        # 用本地构造的模型代替下载，验证传入的权重枚举与替换后的结构。
        backbone = resnet18(weights=None)
        original_body = backbone.layer1[0].conv1.weight
        with patch("leukocyte_analysis.models.resnet.resnet18", return_value=backbone) as constructor:
            model = build_resnet18(5)
        constructor.assert_called_once_with(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.assertIs(model.layer1[0].conv1.weight, original_body)
        self.assertEqual(tuple(model.conv1.weight.shape), (64, 1, 3, 3))
        self.assertIsInstance(model.maxpool, nn.Identity)
        self.assertEqual(model.fc.out_features, 5)

    def test_new_conv_uses_kaiming_initialization(self) -> None:
        backbone = resnet18(weights=None)
        with patch("leukocyte_analysis.models.resnet.resnet18", return_value=backbone):
            with patch("leukocyte_analysis.models.resnet.nn.init.kaiming_normal_") as initialize:
                model = build_resnet18(5, initialization="random")
        initialize.assert_called_once_with(model.conv1.weight, mode="fan_out", nonlinearity="relu")


class TestMetrics(unittest.TestCase):
    def test_perfect_predictions(self) -> None:
        labels = torch.repeat_interleave(torch.arange(5), torch.arange(1, 6))
        score = MulticlassF1Score(num_classes=5, average="macro")(labels, labels)
        recalls = MulticlassRecall(num_classes=5, average=None)(labels, labels)
        self.assertEqual(score.item(), 1.0)
        self.assertTrue(torch.all(recalls == 1))


if __name__ == "__main__":
    unittest.main()
