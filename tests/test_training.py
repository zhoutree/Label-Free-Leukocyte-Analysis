"""训练入口和指标的轻量测试，不加载完整图像数据。"""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import MulticlassF1Score, MulticlassRecall
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import v2

from leukocyte_analysis.data.bbbc045_config import CLASS_NAMES, model_label
from leukocyte_analysis.data.dataset import LeukocyteDataset
from leukocyte_analysis.data.split import split_for_donor, summarize_split_groups
from leukocyte_analysis.models.resnet import build_resnet18
from leukocyte_analysis.cross_validate import parse_args as parse_cv_args
from leukocyte_analysis.train import PROJECT_ROOT, parse_args
from leukocyte_analysis.training.loaders import make_loaders
from leukocyte_analysis.training.module import LeukocyteClassifier
from leukocyte_analysis.training.progress import CompactRichProgressBar


class TestTrainingConfiguration(unittest.TestCase):
    def test_default_data_paths(self) -> None:
        with patch.object(sys, "argv", ["leukocyte-train"]):
            args = parse_args()
        self.assertEqual(args.split_csv, PROJECT_ROOT / "data/processed/bbbc045_split.csv")
        self.assertEqual(args.stats_json, PROJECT_ROOT / "data/processed/bbbc045_train_stats.json")
        self.assertEqual(args.init, "imagenet")
        self.assertEqual(args.lr, 1e-4)
        self.assertEqual(args.weight_decay, 0.01)
        self.assertEqual(args.augmentation, "flips_photometric_geometric")
        self.assertEqual(args.lr_scheduler, "plateau")
        self.assertEqual(args.epochs, 30)
        self.assertEqual(args.early_stopping_patience, 15)
        self.assertEqual(args.label_smoothing, 0.0)
        self.assertFalse(hasattr(args, "task"))

    def test_cross_validation_defaults_and_old_baseline_remain_selectable(self) -> None:
        with patch.object(sys, "argv", ["leukocyte-cross-validate"]):
            args = parse_cv_args()
        self.assertEqual((args.augmentation, args.lr_scheduler),
                         ("flips_photometric_geometric", "plateau"))
        self.assertEqual((args.epochs, args.early_stopping_patience), (30, 15))
        self.assertEqual(args.label_smoothing, 0.0)
        with patch.object(sys, "argv", ["leukocyte-train", "--augmentation", "flips",
                                            "--lr-scheduler", "none"]):
            old = parse_args()
        self.assertEqual((old.augmentation, old.lr_scheduler), ("flips", "none"))

    def test_first_fold_and_continue_options_are_exclusive(self) -> None:
        with patch.object(sys, "argv", ["leukocyte-cross-validate", "--first-fold-only"]):
            self.assertTrue(parse_cv_args().first_fold_only)
        with patch.object(sys, "argv", ["leukocyte-cross-validate", "--continue-run", "run"]):
            self.assertEqual(parse_cv_args().continue_run, Path("run"))
        with patch.object(sys, "argv", ["leukocyte-cross-validate", "--first-fold-only",
                                            "--continue-run", "run"]):
            with patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    parse_cv_args()

    def test_label_smoothing_cli_is_validated(self) -> None:
        with patch.object(sys, "argv", ["leukocyte-cross-validate", "--label-smoothing", "0.05",
                                        "--early-stopping-min-delta", "0"]):
            args = parse_cv_args()
        self.assertEqual((args.label_smoothing, args.early_stopping_min_delta), (0.05, 0.0))
        for invalid in ("-0.1", "1", "nan"):
            with patch.object(sys, "argv", ["leukocyte-cross-validate", "--label-smoothing",
                                            invalid]):
                with patch("sys.stderr"), self.assertRaises(SystemExit):
                    parse_cv_args()

    def test_smoothing_changes_only_training_objective(self) -> None:
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
        self.assertTrue(torch.allclose(model.val_loss.compute(),
                                       F.cross_entropy(logits, targets)))
        self.assertAlmostEqual(model.train_objective_loss.compute().item(), expected.item())
        self.assertAlmostEqual(model.label_smoothing, 0.05)
        with self.assertRaisesRegex(ValueError, "标签平滑"):
            LeukocyteClassifier(backbone, weights, 1e-4, 0.0, label_smoothing=1.0)

    def test_b_and_t_map_to_lymphocyte(self) -> None:
        self.assertEqual(CLASS_NAMES,
                         ("lymphocyte", "eosinophil", "monocyte", "neutrophil"))
        self.assertEqual(model_label("B"), "lymphocyte")
        self.assertEqual(model_label("T"), "lymphocyte")
        with self.assertRaises(ValueError):
            model_label("unknown")

    def test_crf101_years_stay_together_in_regular_training(self) -> None:
        self.assertEqual(split_for_donor("CRF101_2014"), "train")
        self.assertEqual(split_for_donor("CRF101_2015"), "train")
        records = [
            {"donor": "CRF101_2014", "split": "train"},
            {"donor": "CRF101_2015", "split": "train"},
            {"donor": "CRF049", "split": "test"},
        ]
        groups = summarize_split_groups(records)
        self.assertIn({"group": "CRF101", "split": "train",
                       "donors": ["CRF101_2014", "CRF101_2015"], "tiles": 2}, groups)
        records[1]["split"] = "val"
        with self.assertRaisesRegex(ValueError, "CRF101"):
            summarize_split_groups(records)
        records[1]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "CRF101"):
            summarize_split_groups(records)

    def test_dataset_maps_b_and_t_without_rewriting_manifest(self) -> None:
        # 合成清单保持原始 B/T 标签，模型读取时统一合并。
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            split_csv, tiles_npy, stats_json = (root / "split.csv", root / "tiles.npy",
                                                root / "stats.json")
            with split_csv.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=("split", "label", "donor"))
                writer.writeheader()
                for label in ("B", "T", "eosinophil", "neutrophil"):
                    writer.writerow({"split": "train", "label": label, "donor": "synthetic"})
            np.save(tiles_npy, np.zeros((4, 55, 55), dtype=np.float32))
            stats_json.write_text(json.dumps({"mean": 0, "std": 1}), encoding="utf-8")

            dataset = LeukocyteDataset(split_csv, tiles_npy, stats_json, "train")
            self.assertEqual([row[1] for row in dataset.samples], [0, 0, 1, 3])
            self.assertEqual(tuple(dataset[0][0].shape), (1, 55, 55))
            indexed = LeukocyteDataset(split_csv, tiles_npy, stats_json, "train",
                                       include_index=True)
            self.assertEqual(indexed[2][1:], (1, 2))
            # np.load(mmap_mode="r") 在 Windows 上需释放映射才能清理临时文件。
            del dataset, indexed

    def test_raw_transform_precedes_normalization(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            split_csv, tiles_npy, stats_json = (root / "split.csv", root / "tiles.npy",
                                                root / "stats.json")
            with split_csv.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=("split", "label", "donor"))
                writer.writeheader()
                writer.writerow({"split": "train", "label": "B", "donor": "synthetic"})
                writer.writerow({"split": "val", "label": "B", "donor": "synthetic2"})
            np.save(tiles_npy, np.full((2, 55, 55), 2, dtype=np.float32))
            stats_json.write_text(json.dumps({"mean": 2, "std": 2}), encoding="utf-8")

            train = LeukocyteDataset(split_csv, tiles_npy, stats_json, "train",
                                     raw_transform=lambda image: image + 2)
            val = LeukocyteDataset(split_csv, tiles_npy, stats_json, "val")
            self.assertEqual(float(train[0][0].mean()), 1.0)
            self.assertEqual(float(val[0][0].mean()), 0.0)
            self.assertEqual(tuple(train[0][0].shape), (1, 55, 55))
            del train, val

    def test_random_initialization_can_be_selected(self) -> None:
        with patch.object(sys, "argv", ["leukocyte-train", "--init", "random", "--weight-decay", "0"]):
            args = parse_args()
        self.assertEqual(args.init, "random")
        self.assertEqual(args.weight_decay, 0.0)

    def test_only_training_loader_receives_augmentation(self) -> None:
        # 不打开正式数据文件，只核对增强不会进入验证集。
        training_samples = [(index, index, "donor") for index in range(4)]
        training_set = SimpleNamespace(samples=training_samples)
        validation_set = SimpleNamespace(samples=[])
        with patch("leukocyte_analysis.training.loaders.LeukocyteDataset",
                   side_effect=[training_set, validation_set]) as dataset_type:
            with patch("leukocyte_analysis.training.loaders.DataLoader",
                       side_effect=[object(), object()]):
                make_loaders(None, None, None, 8, 42, torch.device("cpu"), "flips")

        transform = dataset_type.call_args_list[0].kwargs["transform"]
        self.assertIsNotNone(transform)
        self.assertNotIn("transform", dataset_type.call_args_list[1].kwargs)
        image = torch.arange(9).reshape(1, 3, 3)
        self.assertEqual(sorted(transform(image).flatten().tolist()), list(range(9)))

    def test_photometric_augmentation_only_reaches_training_set(self) -> None:
        training_set = SimpleNamespace(samples=[(index, index, "donor") for index in range(4)])
        validation_set = SimpleNamespace(samples=[])
        with patch("leukocyte_analysis.training.loaders.LeukocyteDataset",
                   side_effect=[training_set, validation_set]) as dataset_type:
            with patch("leukocyte_analysis.training.loaders.DataLoader",
                       side_effect=[object(), object()]):
                make_loaders(None, None, None, 8, 42, torch.device("cpu"),
                             "flips_photometric")
        train_kwargs = dataset_type.call_args_list[0].kwargs
        val_kwargs = dataset_type.call_args_list[1].kwargs
        self.assertIsNotNone(train_kwargs["transform"])
        self.assertEqual(len(train_kwargs["transform"].transforms), 2)
        self.assertIsNotNone(train_kwargs["raw_transform"])
        self.assertNotIn("raw_transform", val_kwargs)
        self.assertNotIn("transform", val_kwargs)
        transformed = train_kwargs["raw_transform"](torch.full((1, 55, 55), 0.003))
        self.assertEqual(tuple(transformed.shape), (1, 55, 55))
        self.assertEqual(transformed.dtype, torch.float32)

    def test_geometric_augmentation_only_reaches_training_set(self) -> None:
        training_set = SimpleNamespace(samples=[(index, index, "donor") for index in range(4)])
        validation_set = SimpleNamespace(samples=[])
        with patch("leukocyte_analysis.training.loaders.LeukocyteDataset",
                   side_effect=[training_set, validation_set]) as dataset_type:
            with patch("leukocyte_analysis.training.loaders.DataLoader",
                       side_effect=[object(), object()]):
                make_loaders(None, None, None, 8, 42, torch.device("cpu"),
                             "flips_photometric_geometric")
        train_kwargs = dataset_type.call_args_list[0].kwargs
        val_kwargs = dataset_type.call_args_list[1].kwargs
        transforms = train_kwargs["transform"].transforms
        self.assertEqual(len(transforms), 4)
        self.assertIsInstance(transforms[2], v2.RandomChoice)
        self.assertIsInstance(transforms[3], v2.RandomApply)
        affine = transforms[3].transforms[0]
        self.assertIsInstance(affine, v2.RandomAffine)
        self.assertEqual(tuple(affine.scale), (0.95, 1.05))
        raw_transforms = train_kwargs["raw_transform"].transforms
        self.assertEqual(len(raw_transforms), 3)
        self.assertIsInstance(raw_transforms[1].transforms[0], v2.GaussianBlur)
        noise = raw_transforms[2].transforms[0]
        self.assertIsInstance(noise, v2.GaussianNoise)
        self.assertEqual(noise.sigma, 2e-5)
        self.assertFalse(noise.clip)
        self.assertNotIn("transform", val_kwargs)
        self.assertNotIn("raw_transform", val_kwargs)
        image = torch.randn(1, 55, 55)
        for _ in range(8):
            transformed = train_kwargs["transform"](image)
            self.assertEqual(tuple(transformed.shape), (1, 55, 55))
            self.assertTrue(torch.isfinite(transformed).all())
        raw_image = torch.full((1, 55, 55), 0.003)
        for _ in range(8):
            transformed = train_kwargs["raw_transform"](raw_image)
            self.assertEqual(tuple(transformed.shape), (1, 55, 55))
            self.assertTrue(torch.isfinite(transformed).all())

    def test_pretrained_backbone_and_small_image_layers(self) -> None:
        # 用本地构造的模型代替下载，验证传入的权重枚举与替换后的结构。
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
        backbone = resnet18(weights=None)
        with patch("leukocyte_analysis.models.resnet.resnet18", return_value=backbone):
            with patch("leukocyte_analysis.models.resnet.nn.init.kaiming_normal_") as initialize:
                model = build_resnet18(4, initialization="random")
        initialize.assert_called_once_with(model.conv1.weight, mode="fan_out", nonlinearity="relu")

    def test_rich_progress_adapts_metrics_to_terminal_width(self) -> None:
        # 只改变终端显示，不改变日志中的完整指标名称。
        reported = {"step_weighted_loss": 0.4, "val_loss": 0.2,
                    "val_macro_f1": 0.8, "v_num": 1}
        with patch("leukocyte_analysis.training.progress.RichProgressBar.get_metrics",
                   return_value=reported):
            progress = CompactRichProgressBar(refresh_rate=5)
            progress._console = SimpleNamespace(width=160)
            self.assertEqual(progress.get_metrics(None, None),
                             {"loss": 0.4, "val_loss": 0.2, "val_f1": 0.8})
            progress._console.width = 125
            self.assertEqual(progress.get_metrics(None, None), {"loss": 0.4, "val_f1": 0.8})
            progress._console.width = 95
            self.assertEqual(progress.get_metrics(None, None), {"loss": 0.4})
            progress._console.width = 60
            self.assertEqual(progress.get_metrics(None, None), {})
            self.assertEqual(str(progress.configure_columns(None)[-1].render(None)), "")


class TestMetrics(unittest.TestCase):
    def test_perfect_predictions(self) -> None:
        labels = torch.repeat_interleave(torch.arange(4), torch.arange(1, 5))
        score = MulticlassF1Score(num_classes=4, average="macro")(labels, labels)
        recalls = MulticlassRecall(num_classes=4, average=None)(labels, labels)
        self.assertEqual(score.item(), 1.0)
        self.assertTrue(torch.all(recalls == 1))


if __name__ == "__main__":
    unittest.main()
