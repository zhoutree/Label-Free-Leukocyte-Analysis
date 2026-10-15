"""验证训练增强配置与验证集隔离，不读取正式图块。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torchvision.transforms import v2

from leukocyte_analysis.training.loaders import make_loaders


class TestAugmentation(unittest.TestCase):
    """确认随机增强只进入训练集，并保持图块形状和有限值。"""

    def _loader_options(self, augmentation: str) -> dict:
        """捕获传给数据集的增强参数，并统一核对验证集不受扰动。"""
        training_set = SimpleNamespace(samples=[(index, index, "donor") for index in range(4)])
        validation_set = SimpleNamespace(samples=[])
        with patch("leukocyte_analysis.training.loaders.LeukocyteDataset",
                   side_effect=[training_set, validation_set]) as dataset_type:
            with patch("leukocyte_analysis.training.loaders.DataLoader",
                       side_effect=[object(), object()]):
                make_loaders(None, None, None, 8, 42, torch.device("cpu"), augmentation)
        train_kwargs = dataset_type.call_args_list[0].kwargs
        val_kwargs = dataset_type.call_args_list[1].kwargs
        self.assertNotIn("transform", val_kwargs)
        self.assertNotIn("raw_transform", val_kwargs)
        return train_kwargs

    def test_only_training_loader_receives_augmentation(self) -> None:
        """纯翻转模式不得改变图块像素集合。"""
        options = self._loader_options("flips")
        transform = options["transform"]
        self.assertIsNotNone(transform)
        image = torch.arange(9).reshape(1, 3, 3)
        self.assertEqual(sorted(transform(image).flatten().tolist()), list(range(9)))

    def test_photometric_augmentation_only_reaches_training_set(self) -> None:
        """亮度／对比度扰动作用于原始灰度，不泄漏到验证集。"""
        options = self._loader_options("flips_photometric")
        self.assertIsNotNone(options["transform"])
        self.assertEqual(len(options["transform"].transforms), 2)
        self.assertIsNotNone(options["raw_transform"])
        transformed = options["raw_transform"](torch.full((1, 55, 55), 0.003))
        self.assertEqual(tuple(transformed.shape), (1, 55, 55))
        self.assertEqual(transformed.dtype, torch.float32)

    def test_geometric_augmentation_only_reaches_training_set(self) -> None:
        """完整增强包含保守几何、模糊与噪声且输出仍可训练。"""
        options = self._loader_options("flips_photometric_geometric")
        transforms = options["transform"].transforms
        self.assertEqual(len(transforms), 4)
        self.assertIsInstance(transforms[2], v2.RandomChoice)
        self.assertIsInstance(transforms[3], v2.RandomApply)
        affine = transforms[3].transforms[0]
        self.assertIsInstance(affine, v2.RandomAffine)
        self.assertEqual(tuple(affine.scale), (0.95, 1.05))
        raw_transforms = options["raw_transform"].transforms
        self.assertEqual(len(raw_transforms), 3)
        self.assertIsInstance(raw_transforms[1].transforms[0], v2.GaussianBlur)
        noise = raw_transforms[2].transforms[0]
        self.assertIsInstance(noise, v2.GaussianNoise)
        self.assertEqual(noise.sigma, 2e-5)
        self.assertFalse(noise.clip)
        image = torch.randn(1, 55, 55)
        for _ in range(8):
            transformed = options["transform"](image)
            self.assertEqual(tuple(transformed.shape), (1, 55, 55))
            self.assertTrue(torch.isfinite(transformed).all())
        raw_image = torch.full((1, 55, 55), 0.003)
        for _ in range(8):
            transformed = options["raw_transform"](raw_image)
            self.assertEqual(tuple(transformed.shape), (1, 55, 55))
            self.assertTrue(torch.isfinite(transformed).all())
