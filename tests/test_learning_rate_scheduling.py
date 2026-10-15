"""验证学习率调度器与早停回调的配置和训练步进。"""

import unittest
from unittest.mock import patch

import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from torch import nn

from leukocyte_analysis.training.experiment import run_training
from leukocyte_analysis.training.module import LeukocyteClassifier
from _experiment_fixtures import four_sample_loader, temporary_run_root


class TestLearningRateScheduling(unittest.TestCase):
    """检查 plateau 与 cosine 调度器及早停策略各自的行为。"""

    def test_early_stopping_uses_the_checkpoint_metric(self) -> None:
        """Plateau 和早停都监控用于选择检查点的验证 macro-F1。"""
        loader = four_sample_loader()
        with temporary_run_root() as root:
            with patch("leukocyte_analysis.training.experiment.L.Trainer") as trainer:
                run_training(nn.Linear(1, 4), loader, loader, torch.ones(4),
                             torch.device("cpu"), 20, root / "run",
                             1e-4, 0.01, early_stopping_patience=15,
                             lr_scheduler="plateau")
        callbacks = trainer.call_args.kwargs["callbacks"]
        early_stop = next(item for item in callbacks if isinstance(item, EarlyStopping))
        self.assertEqual(early_stop.monitor, "val_macro_f1")
        self.assertEqual(early_stop.patience, 15)
        self.assertTrue(any(isinstance(item, LearningRateMonitor) for item in callbacks))
        module = trainer.return_value.fit.call_args.args[0]
        configured = module.configure_optimizers()
        self.assertEqual(configured["lr_scheduler"]["monitor"], "val_macro_f1")
        scheduler = configured["lr_scheduler"]["scheduler"]
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        self.assertEqual((scheduler.patience, scheduler.cooldown), (3, 2))
        for _ in range(5):
            scheduler.step(0.5)
        self.assertAlmostEqual(configured["optimizer"].param_groups[0]["lr"], 5e-5)

    def test_cosine_scheduler_and_early_stopping_are_independent(self) -> None:
        """余弦调度按轮次推进，不从噪声验证指标触发降率。"""
        loader = four_sample_loader()
        with temporary_run_root() as root:
            with patch("leukocyte_analysis.training.experiment.L.Trainer") as trainer:
                run_training(nn.Linear(1, 4), loader, loader, torch.ones(4),
                             torch.device("cpu"), 30, root / "run",
                             1e-4, 0.01, early_stopping_patience=10,
                             lr_scheduler="cosine")
        callbacks = trainer.call_args.kwargs["callbacks"]
        early_stop = next(item for item in callbacks if isinstance(item, EarlyStopping))
        self.assertEqual((early_stop.monitor, early_stop.patience), ("val_macro_f1", 10))
        self.assertEqual(early_stop.min_delta, 1e-3)
        self.assertEqual(trainer.call_args.kwargs["min_epochs"], 15)
        module = trainer.return_value.fit.call_args.args[0]
        configured = module.configure_optimizers()
        self.assertNotIn("monitor", configured["lr_scheduler"])
        self.assertIsInstance(configured["lr_scheduler"]["scheduler"],
                              torch.optim.lr_scheduler.SequentialLR)
        self.assertAlmostEqual(configured["optimizer"].param_groups[0]["lr"], 5e-5)

    def test_cosine_warmup_decay_and_resume_tail(self) -> None:
        """暖启动后降至下限，超过原日程的续训不会反弹。"""
        module = LeukocyteClassifier(nn.Linear(1, 4), torch.ones(4), 1e-4, 0.0,
                                     lr_scheduler="cosine", schedule_epochs=5)
        configured = module.configure_optimizers()
        optimizer = configured["optimizer"]
        scheduler = configured["lr_scheduler"]["scheduler"]
        rates = [optimizer.param_groups[0]["lr"]]
        for _ in range(8):
            optimizer.step()
            scheduler.step()
            rates.append(optimizer.param_groups[0]["lr"])
        self.assertAlmostEqual(rates[0], 5e-5)
        self.assertAlmostEqual(rates[2], 1e-4)
        self.assertTrue(all(left >= right for left, right in zip(rates[2:5], rates[3:6])))
        self.assertTrue(all(abs(rate - 1e-6) < 1e-10 for rate in rates[5:]))


if __name__ == "__main__":
    unittest.main()
