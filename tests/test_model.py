"""验证小图 ResNet、分类器损失和带索引批次的预测接口。"""

import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18

from leukocyte_analysis.data.bbbc045_config import CLASS_NAMES
from leukocyte_analysis.models.resnet import build_resnet18
from leukocyte_analysis.training.module import LeukocyteClassifier


class TestModel(unittest.TestCase):
    """约束小图 ResNet 结构、分类损失口径和推理批次格式。"""

    def test_indexed_batch_can_be_predicted(self) -> None:
        """推理步骤接受附带原清单索引的验证批次。"""
        model = LeukocyteClassifier(nn.Flatten(start_dim=1), torch.ones(4), 1e-4, 0.0)
        images = torch.tensor([[[[0.0, 1.0, 0.0, 0.0]]]])
        targets = torch.tensor([1])
        result = model.predict_step((images, targets, torch.tensor([123])), 0)
        self.assertEqual(result[0].tolist(), [1])
        self.assertEqual(result[1].tolist(), [1])

    def test_smoothing_changes_only_training_objective(self) -> None:
        """标签平滑只改变训练目标，不改未平滑的验证损失。"""
        backbone = nn.Linear(2, len(CLASS_NAMES), bias=False)
        with torch.no_grad():
            backbone.weight.copy_(torch.arange(8, dtype=torch.float32).reshape(4, 2) / 10)
        weights = torch.tensor([1.0, 2.0, 3.0, 4.0])
        images = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        targets = torch.tensor([0, 2])
        model = LeukocyteClassifier(backbone, weights, 1e-4, 0.0, label_smoothing=0.05)
        with patch.object(model, "log"):
            actual = model.training_step((images, targets), 0)
            model.validation_step((images, targets), 0)
        logits = backbone(images)
        expected = F.cross_entropy(logits, targets, weight=weights, label_smoothing=0.05)
        self.assertTrue(torch.allclose(actual, expected))
        # 验证损失保持未平滑，以便与旧实验按同一口径比较。
        self.assertTrue(torch.allclose(model.val_loss.compute(),
                                       F.cross_entropy(logits, targets)))
        self.assertAlmostEqual(model.train_objective_loss.compute().item(), expected.item())
        self.assertAlmostEqual(model.label_smoothing, 0.05)
        with self.assertRaisesRegex(ValueError, "标签平滑"):
            LeukocyteClassifier(backbone, weights, 1e-4, 0.0, label_smoothing=1.0)

    def test_pretrained_backbone_and_small_image_layers(self) -> None:
        """保留预训练主体并把输入层改为适合 55×55 灰度图。"""
        # 用本地模型核对预训练权重枚举，避免单元测试下载 ImageNet 权重。
        backbone = resnet18(weights=None)
        original_body = backbone.layer1[0].conv1.weight
        with patch("leukocyte_analysis.models.resnet.resnet18", return_value=backbone) as constructor:
            model = build_resnet18(len(CLASS_NAMES))
        constructor.assert_called_once_with(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.assertIs(model.layer1[0].conv1.weight, original_body)
        self.assertEqual(tuple(model.conv1.weight.shape), (64, 1, 3, 3))
        self.assertIsInstance(model.maxpool, nn.Identity)
        self.assertEqual(model.fc.out_features, 4)

    def test_new_conv_uses_kaiming_initialization(self) -> None:
        """从头训练时新建的首层卷积使用 Kaiming 初始化。"""
        backbone = resnet18(weights=None)
        with patch("leukocyte_analysis.models.resnet.resnet18", return_value=backbone):
            with patch("leukocyte_analysis.models.resnet.nn.init.kaiming_normal_") as initialize:
                model = build_resnet18(4, initialization="random")
        initialize.assert_called_once_with(model.conv1.weight, mode="fan_out", nonlinearity="relu")
