"""配置 Lightning 训练器、标准日志和最佳模型检查点。"""

import json
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from torch import nn
from torch.utils.data import DataLoader

from .error_logging import ValidationErrorLogger
from .module import LeukocyteClassifier
from .progress import CompactRichProgressBar, prepare_console_output


def run_training(
    backbone: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    epochs: int,
    output_dir: Path,
    learning_rate: float,
    weight_decay: float,
    smoke_test: bool = False,
    early_stopping_patience: int = 0,
    lr_scheduler: str = "none",
    resume_checkpoint: Path | None = None,
    data_split_record: dict | None = None,
) -> None:
    """按验证 macro-F1 选最佳权重；可提前停止，冒烟模式不保存文件。"""
    if early_stopping_patience < 0:
        raise ValueError("early_stopping_patience 不得小于 0")
    if resume_checkpoint is not None and (smoke_test or not resume_checkpoint.is_file()):
        raise ValueError("续训需要现存检查点，且不能使用冒烟模式")
    if resume_checkpoint is not None and any(
        (output_dir / name).exists() for name in ("csv", "tensorboard", "checkpoints")
    ):
        raise FileExistsError("续训输出目录已有训练日志或检查点")
    prepare_console_output()
    module = LeukocyteClassifier(backbone, class_weights, learning_rate, weight_decay,
                                 lr_scheduler=lr_scheduler)
    checkpoint = None
    # 使用 Lightning 的 Rich 进度条按终端宽度排版；限制刷新频率以减少重绘。
    callbacks = [CompactRichProgressBar(refresh_rate=5)]
    if smoke_test:
        loggers = False
    else:
        output_dir.mkdir(parents=True, exist_ok=resume_checkpoint is not None)
        if data_split_record is not None:
            (output_dir / "data_split.json").write_text(
                json.dumps(data_split_record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        # 两种日志由 Lightning 管理，不再手写 CSV 和 TensorBoard 事件。
        loggers = [
            TensorBoardLogger(output_dir, name="tensorboard", version="", default_hp_metric=False),
            CSVLogger(output_dir, name="csv", version=""),
        ]
        checkpoint = ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="best",
            monitor="val_macro_f1",
            mode="max",
            save_top_k=1,
            save_last=True,
            save_on_exception=True,
            enable_version_counter=False,
        )
        callbacks.append(checkpoint)
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))
        if getattr(val_loader.dataset, "include_index", False):
            # Lightning 回调记录每轮错分，稳定索引由验证数据集提供。
            callbacks.append(ValidationErrorLogger(
                val_loader.dataset, output_dir / "validation_errors"
            ))
        if early_stopping_patience > 0:
            # 与最佳权重使用同一验证指标；连续若干次无提升后停止。
            callbacks.append(EarlyStopping(
                monitor="val_macro_f1", mode="max",
                patience=early_stopping_patience, min_delta=0.0,
            ))

    trainer = L.Trainer(
        accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=1,
        max_epochs=epochs,
        default_root_dir=output_dir,
        logger=loggers,
        callbacks=callbacks,
        enable_checkpointing=not smoke_test,
        fast_dev_run=2 if smoke_test else False,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        enable_model_summary=False,
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader,
                ckpt_path=str(resume_checkpoint) if resume_checkpoint else None)
    if smoke_test:
        print("冒烟测试通过：训练和验证各执行了两批，未保存权重。")
    else:
        print("最佳验证 macro-F1：", checkpoint.best_model_score,
              "检查点：", checkpoint.best_model_path)
