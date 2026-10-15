"""训练和交叉验证命令行的默认配置与参数约束。"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from leukocyte_analysis.cross_validate import parse_args as parse_cv_args
from leukocyte_analysis.train import PROJECT_ROOT, parse_args


class TestTrainingConfiguration(unittest.TestCase):
    """防止默认方案或历史兼容入口在重构时被无意改动。"""

    def test_default_data_paths(self) -> None:
        """正式训练默认使用当前四分类数据及增强、调度设置。"""
        with patch.object(sys, "argv", ["leukocyte-train"]):
            args = parse_args()
        self.assertEqual(args.split_csv, PROJECT_ROOT / "data/processed/bbbc045_split.csv")
        self.assertEqual(args.stats_json, PROJECT_ROOT / "data/processed/bbbc045_train_stats.json")
        self.assertEqual(args.init, "imagenet")
        self.assertEqual(args.lr, 1e-4)
        self.assertEqual(args.weight_decay, 0.01)
        self.assertEqual(args.augmentation, "flips_photometric_geometric")
        self.assertEqual(args.lr_scheduler, "plateau")
        self.assertEqual(args.epochs, 45)
        self.assertEqual(args.early_stopping_patience, 15)
        self.assertEqual(args.early_stopping_min_delta, 0.0)
        self.assertEqual(args.label_smoothing, 0.05)
        self.assertFalse(hasattr(args, "task"))

    def test_cross_validation_defaults_and_old_baseline_remain_selectable(self) -> None:
        """交叉验证与常规训练默认一致，旧翻转基线仍可显式复现。"""
        with patch.object(sys, "argv", ["leukocyte-cross-validate"]):
            args = parse_cv_args()
        self.assertEqual((args.augmentation, args.lr_scheduler),
                         ("flips_photometric_geometric", "plateau"))
        self.assertEqual((args.epochs, args.early_stopping_patience), (30, 15))
        self.assertEqual(args.early_stopping_min_delta, 0.0)
        self.assertEqual(args.label_smoothing, 0.05)
        with patch.object(sys, "argv", ["leukocyte-train", "--augmentation", "flips",
                                            "--lr-scheduler", "none"]):
            old = parse_args()
        self.assertEqual((old.augmentation, old.lr_scheduler), ("flips", "none"))

    def test_latest_defaults_can_be_overridden(self) -> None:
        """训练轮数、学习率和平滑系数仍由命令行显式选择。"""
        for command, parse in (("leukocyte-train", parse_args),
                               ("leukocyte-cross-validate", parse_cv_args)):
            with self.subTest(command=command), patch.object(
                sys, "argv", [command, "--epochs", "60", "--lr", "0.00005",
                              "--label-smoothing", "0"],
            ):
                args = parse()
            self.assertEqual((args.epochs, args.lr, args.label_smoothing),
                             (60, 5e-5, 0.0))

    def test_first_fold_and_continue_options_are_exclusive(self) -> None:
        """首折试跑与接续已有运行不可在同一命令中混用。"""
        with patch.object(sys, "argv", ["leukocyte-cross-validate", "--first-fold-only"]):
            self.assertTrue(parse_cv_args().first_fold_only)
        with patch.object(sys, "argv", ["leukocyte-cross-validate", "--continue-run", "run"]):
            self.assertEqual(parse_cv_args().continue_run, Path("run"))
        with patch.object(sys, "argv", ["leukocyte-cross-validate", "--first-fold-only",
                                            "--continue-run", "run"]):
            with patch("sys.stderr"), self.assertRaises(SystemExit):
                parse_cv_args()

    def test_label_smoothing_cli_is_validated(self) -> None:
        """拒绝超出交叉熵允许范围或非有限的平滑系数。"""
        with patch.object(sys, "argv", ["leukocyte-cross-validate", "--label-smoothing", "0.05",
                                        "--early-stopping-min-delta", "0"]):
            args = parse_cv_args()
        self.assertEqual((args.label_smoothing, args.early_stopping_min_delta), (0.05, 0.0))
        for invalid in ("-0.1", "1", "nan"):
            with patch.object(sys, "argv", ["leukocyte-cross-validate", "--label-smoothing",
                                            invalid]):
                with patch("sys.stderr"), self.assertRaises(SystemExit):
                    parse_cv_args()

    def test_shared_training_validation_rejects_invalid_lr(self) -> None:
        """两种入口都通过同一检查拒绝零学习率。"""
        for command, parse in (("leukocyte-train", parse_args),
                               ("leukocyte-cross-validate", parse_cv_args)):
            with self.subTest(command=command), patch.object(sys, "argv", [command, "--lr", "0"]):
                with patch("sys.stderr"), self.assertRaises(SystemExit):
                    parse()

    def test_random_initialization_can_be_selected(self) -> None:
        """保留从头训练和关闭权重衰减的显式选择。"""
        with patch.object(sys, "argv", ["leukocyte-train", "--init", "random", "--weight-decay", "0"]):
            args = parse_args()
        self.assertEqual(args.init, "random")
        self.assertEqual(args.weight_decay, 0.0)
