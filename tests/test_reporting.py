"""用少量合成样本验证 Lightning 的训练日志和检查点。"""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from leukocyte_analysis.data.bbbc045_config import CLASS_NAMES
from leukocyte_analysis.training.experiment import run_training
from leukocyte_analysis.training.error_logging import ValidationErrorLogger
from leukocyte_analysis.evaluation.runner import load_classifier


class TestReporting(unittest.TestCase):
    def test_regular_training_records_donor_grouping(self) -> None:
        images = torch.zeros(4, 1)
        targets = torch.arange(4)
        loader = DataLoader(TensorDataset(images, targets), batch_size=4)
        record = {"group_overrides": {"CRF101_2014": "CRF101"},
                  "groups": [{"group": "CRF101", "split": "train",
                              "donors": ["CRF101_2014", "CRF101_2015"], "tiles": 2}]}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            output = Path(temporary) / "run"
            with patch("leukocyte_analysis.training.experiment.L.Trainer"):
                run_training(nn.Linear(1, 4), loader, loader, torch.ones(4),
                             torch.device("cpu"), 1, output, 1e-4, 0.0,
                             data_split_record=record)
            self.assertEqual(json.loads((output / "data_split.json").read_text(
                encoding="utf-8")), record)

    def test_four_class_module_uses_lymphocyte_metric(self) -> None:
        from leukocyte_analysis.training.module import LeukocyteClassifier

        module = LeukocyteClassifier(nn.Linear(1, 4), torch.ones(4), 1e-4, 0.0)
        self.assertEqual(module.class_names, CLASS_NAMES)
        self.assertEqual(module.val_f1.num_classes, 4)

    def test_four_class_checkpoint_restores_weights(self) -> None:
        # 用合成数据检查四分类保存与恢复，不接触保留测试集。
        images = torch.arange(8, dtype=torch.float32).reshape(-1, 1) / 8
        targets = torch.arange(8) % 4
        loader = DataLoader(TensorDataset(images, targets), batch_size=4)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            output = Path(temporary) / "four_class"
            run_training(nn.Linear(1, 4), loader, loader, torch.ones(4),
                         torch.device("cpu"), 1, output, 0.01, 0.0)
            checkpoint = output / "checkpoints" / "best.ckpt"
            restored = load_classifier(checkpoint, nn.Linear(1, 4))
            self.assertEqual(restored.class_names, CLASS_NAMES)

    def test_early_stopping_uses_the_checkpoint_metric(self) -> None:
        # 仅检查 Lightning 的回调配置，不启动完整训练。
        images = torch.zeros(4, 1)
        targets = torch.arange(4)
        loader = DataLoader(TensorDataset(images, targets), batch_size=4)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            with patch("leukocyte_analysis.training.experiment.L.Trainer") as trainer:
                run_training(nn.Linear(1, 4), loader, loader, torch.ones(4),
                             torch.device("cpu"), 20, Path(temporary) / "run",
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
        images = torch.zeros(4, 1)
        targets = torch.arange(4)
        loader = DataLoader(TensorDataset(images, targets), batch_size=4)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            with patch("leukocyte_analysis.training.experiment.L.Trainer") as trainer:
                run_training(nn.Linear(1, 4), loader, loader, torch.ones(4),
                             torch.device("cpu"), 30, Path(temporary) / "run",
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
        from leukocyte_analysis.training.module import LeukocyteClassifier

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

    def test_resume_passes_checkpoint_to_lightning(self) -> None:
        images = torch.zeros(4, 1)
        targets = torch.arange(4)
        loader = DataLoader(TensorDataset(images, targets), batch_size=4)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
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
        images = torch.arange(8, dtype=torch.float32).reshape(-1, 1) / 8
        targets = torch.arange(8) % 4
        loader = DataLoader(TensorDataset(images, targets), batch_size=4)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
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
        images = torch.arange(8, dtype=torch.float32).reshape(-1, 1) / 8
        targets = torch.arange(8) % 4
        loader = DataLoader(TensorDataset(images, targets), batch_size=4)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
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

    def test_indexed_validation_loader_gets_error_callback(self) -> None:
        images = torch.zeros(4, 1)
        targets = torch.arange(4)
        train_loader = DataLoader(TensorDataset(images, targets), batch_size=4)
        val_dataset = TensorDataset(images, targets)
        val_dataset.include_index = True
        val_loader = DataLoader(val_dataset, batch_size=4)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            with patch("leukocyte_analysis.training.experiment.L.Trainer") as trainer:
                run_training(nn.Linear(1, 4), train_loader, val_loader,
                             torch.ones(4), torch.device("cpu"), 1,
                             Path(temporary) / "run", 0.01, 0.0)
        callbacks = trainer.call_args.kwargs["callbacks"]
        self.assertTrue(any(isinstance(item, ValidationErrorLogger) for item in callbacks))

    def test_indexed_validation_run_writes_error_records(self) -> None:
        # 真实 Lightning 循环 + 合成数据，核对回调和模型步骤能协同工作。
        images = torch.zeros(4, 1)
        targets = torch.arange(4)
        train_loader = DataLoader(TensorDataset(images, targets), batch_size=4)
        val_dataset = TensorDataset(images, targets, torch.arange(4))
        val_dataset.include_index = True
        val_dataset.class_names = CLASS_NAMES
        raw_labels = ("B", "eosinophil", "monocyte", "neutrophil")
        val_dataset.source_by_index = {
            index: {"file": f"synthetic/{label}/batch1/ch5_1.tif",
                    "donor": "synthetic", "source_label": label,
                    "row": index, "col": 0}
            for index, label in enumerate(raw_labels)
        }
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            output = Path(temporary) / "run"
            run_training(nn.Linear(1, 4), train_loader,
                         DataLoader(val_dataset, batch_size=4), torch.ones(4),
                         torch.device("cpu"), 1, output, 0.01, 0.0)
            error_dir = output / "validation_errors"
            self.assertTrue((error_dir / "epoch_000.csv").exists())
            self.assertTrue((error_dir / "frequency.csv").exists())
            with (error_dir / "epoch_000.csv").open(newline="", encoding="utf-8") as handle:
                errors = list(csv.DictReader(handle))
            self.assertGreater(len(errors), 0)
            self.assertTrue(all(row["split"] == "val" for row in errors))

    def test_one_epoch_writes_comparable_losses_and_curves(self) -> None:
        # 四类各两个样本，避免借用正式数据或启动完整 ResNet 训练。
        images = torch.arange(8, dtype=torch.float32).reshape(-1, 1) / 8
        targets = torch.arange(8) % 4
        loader = DataLoader(TensorDataset(images, targets), batch_size=4)
        model = nn.Linear(1, 4)
        weights = torch.tensor([1.0, 2.0, 1.0, 1.0])

        # 使用项目内临时目录，兼容只允许写入工作区的运行环境。
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            output = Path(temporary) / "run"
            run_training(model, loader, loader, weights, torch.device("cpu"),
                         1, output, 0.01, 0.0, lr_scheduler="cosine")

            with (output / "csv" / "metrics.csv").open(newline="", encoding="utf-8") as history:
                rows = list(csv.DictReader(history))
            self.assertTrue(any(row.get("train_weighted_loss") for row in rows))
            self.assertTrue(any(row.get("val_macro_f1") for row in rows))
            self.assertTrue(any(row.get("lr-AdamW") for row in rows))
            self.assertTrue((output / "checkpoints" / "best.ckpt").exists())
            self.assertTrue((output / "checkpoints" / "last.ckpt").exists())

            # 被忽略的模型结构在加载时显式提供；确认检查点能独立恢复。
            restored = load_classifier(output / "checkpoints" / "best.ckpt", nn.Linear(1, 4))
            self.assertEqual(tuple(restored(images).shape), (8, 4))

            events = EventAccumulator(str(output / "tensorboard"))
            events.Reload()
            for name in ("train_loss", "val_loss", "train_weighted_loss",
                         "val_macro_f1", "val_recall_lymphocyte"):
                self.assertEqual(len(events.Scalars(name)), 1)


if __name__ == "__main__":
    unittest.main()
