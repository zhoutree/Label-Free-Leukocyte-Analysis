"""验证 Lightning 检查点续训及跨目录延续训练的结果。"""

import csv
import unittest
from unittest.mock import patch

import torch
from torch import nn

from leukocyte_analysis.training.experiment import run_training
from _experiment_fixtures import (
    balanced_four_class_loader,
    four_sample_loader,
    temporary_run_root,
)


class TestTrainingResume(unittest.TestCase):
    """检查续训参数、原检查点保护和余弦日程尾段。"""

    def test_resume_passes_checkpoint_to_lightning(self) -> None:
        """续训入口把原 last.ckpt 交给 Lightning 恢复优化状态。"""
        loader = four_sample_loader()
        with temporary_run_root() as root:
            checkpoint = root / "last.ckpt"
            checkpoint.touch()
            with patch("leukocyte_analysis.training.experiment.L.Trainer") as trainer:
                run_training(nn.Linear(1, 4), loader, loader, torch.ones(4),
                             torch.device("cpu"), 30, root / "continued", 1e-4, 0.01,
                             early_stopping_patience=5, lr_scheduler="plateau",
                             resume_checkpoint=checkpoint)
            self.assertEqual(trainer.return_value.fit.call_args.kwargs["ckpt_path"],
                             str(checkpoint))

    def test_resume_into_new_directory_keeps_original_checkpoint(self) -> None:
        """跨目录续训保留原检查点，并从下一轮追加指标。"""
        loader = balanced_four_class_loader()
        with temporary_run_root() as root:
            first = root / "first"
            continued = root / "continued"
            run_training(nn.Linear(1, 4), loader, loader, torch.ones(4),
                         torch.device("cpu"), 1, first, 0.01, 0.0,
                         early_stopping_patience=5, lr_scheduler="plateau")
            last = first / "checkpoints" / "last.ckpt"
            original = last.read_bytes()
            run_training(nn.Linear(1, 4), loader, loader, torch.ones(4),
                         torch.device("cpu"), 2, continued, 0.01, 0.0,
                         early_stopping_patience=5, lr_scheduler="plateau",
                         resume_checkpoint=last)
            self.assertEqual(last.read_bytes(), original)
            self.assertTrue((continued / "checkpoints" / "last.ckpt").is_file())
            with (continued / "csv" / "metrics.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                self.assertIn("1", {row["epoch"] for row in csv.DictReader(handle)
                                    if row["val_macro_f1"]})

    def test_cosine_resume_keeps_floor_after_original_horizon(self) -> None:
        """超过原余弦日程终点后仍维持最低学习率。"""
        loader = balanced_four_class_loader()
        with temporary_run_root() as root:
            run_training(nn.Linear(1, 4), loader, loader, torch.ones(4),
                         torch.device("cpu"), 2, root / "first", 1e-4, 0.0,
                         lr_scheduler="cosine")
            last = root / "first" / "checkpoints" / "last.ckpt"
            run_training(nn.Linear(1, 4), loader, loader, torch.ones(4),
                         torch.device("cpu"), 3, root / "continued", 1e-4, 0.0,
                         lr_scheduler="cosine", schedule_epochs=2,
                         resume_checkpoint=last)
            continued = torch.load(root / "continued" / "checkpoints" / "last.ckpt",
                                   map_location="cpu", weights_only=False)
            self.assertAlmostEqual(
                continued["optimizer_states"][0]["param_groups"][0]["lr"], 1e-6,
            )


if __name__ == "__main__":
    unittest.main()
