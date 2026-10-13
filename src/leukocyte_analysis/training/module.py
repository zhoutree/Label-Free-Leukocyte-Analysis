"""使用 Lightning 管理优化步骤，使用 TorchMetrics 汇总整轮指标。"""

import math

import lightning as L
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassF1Score, MulticlassRecall

from ..data.bbbc045_config import CLASS_NAMES


PLATEAU_FACTOR = 0.5
PLATEAU_PATIENCE = 3
PLATEAU_COOLDOWN = 2
PLATEAU_MIN_LR_RATIO = 0.01
COSINE_WARMUP_EPOCHS = 2
COSINE_WARMUP_START_FACTOR = 0.5
COSINE_MIN_LR_RATIO = 0.01


class LeukocyteClassifier(L.LightningModule):
    """保留项目特有的加权损失；交由 Lightning 管理循环与设备。"""

    def __init__(
        self,
        backbone: nn.Module,
        class_weights: torch.Tensor,
        learning_rate: float,
        weight_decay: float,
        lr_scheduler: str = "none",
        schedule_epochs: int = 30,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if len(CLASS_NAMES) != len(class_weights):
            raise ValueError("类别名称数量与类别权重数量不一致")
        if lr_scheduler not in {"none", "plateau", "cosine"}:
            raise ValueError(f"未知学习率调度方式：{lr_scheduler}")
        if schedule_epochs < 1:
            raise ValueError("学习率计划轮数必须大于 0")
        if not math.isfinite(label_smoothing) or not 0 <= label_smoothing < 1:
            raise ValueError("标签平滑系数必须在 [0, 1) 内")
        self.save_hyperparameters(ignore=("backbone", "class_weights"))
        self.backbone = backbone
        self.class_names = CLASS_NAMES
        self.register_buffer("class_weights", class_weights.detach().clone())
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.schedule_epochs = schedule_epochs
        self.label_smoothing = label_smoothing

        self.train_loss = MeanMetric()
        self.train_weighted_loss = MeanMetric()
        self.train_objective_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_f1 = MulticlassF1Score(num_classes=len(self.class_names), average="macro")
        self.val_recall = MulticlassRecall(num_classes=len(self.class_names), average=None)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """返回每类的原始分数，不在模型内做 softmax。"""
        return self.backbone(images)

    def predict_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """供 Trainer.predict 使用；兼容带原始索引的验证批次。"""
        images, targets = batch[:2]
        confidence, predicted = self(images).softmax(dim=1).max(dim=1)
        # 及时转回 CPU，避免 Lightning 汇集全部预测时占用显存。
        return targets.cpu(), predicted.cpu(), confidence.cpu()

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """加权 CE 更新参数；未加权 CE 用于和验证曲线比较。"""
        images, targets = batch
        logits = self(images)
        objective = F.cross_entropy(logits, targets, weight=self.class_weights,
                                    label_smoothing=self.label_smoothing)
        per_cell_loss = F.cross_entropy(logits.detach(), targets, reduction="none")

        # 前两项保留未平滑口径，便于与旧实验比较；另记录实际优化目标。
        self.train_loss.update(per_cell_loss)
        self.train_weighted_loss.update(per_cell_loss, weight=self.class_weights[targets])
        self.train_objective_loss.update(objective.detach(), weight=len(targets))
        self.log("train_loss", self.train_loss, on_step=False, on_epoch=True)
        self.log("train_weighted_loss", self.train_weighted_loss, on_step=False, on_epoch=True)
        self.log("train_objective_loss", self.train_objective_loss,
                 on_step=False, on_epoch=True)
        self.log("step_weighted_loss", objective.detach(), on_step=True, on_epoch=False,
                 prog_bar=True, logger=False)
        return objective

    def validation_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> dict[str, list] | None:
        """验证集保持原始分布；指标只在整轮结束后汇总。"""
        images, targets = batch[:2]
        logits = self(images)
        self.val_loss.update(F.cross_entropy(logits, targets, reduction="none"))
        self.val_f1.update(logits, targets)
        self.val_recall.update(logits, targets)
        self.log("val_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_macro_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

        # 仅把错分图块的小量元数据交给回调；不保存整批 logits 或图像。
        if len(batch) == 3:
            probabilities = logits.detach().softmax(dim=1)
            confidence, predicted = probabilities.max(dim=1)
            wrong = predicted != targets
            return {
                "packed_index": batch[2][wrong].cpu().tolist(),
                "true_index": targets[wrong].cpu().tolist(),
                "predicted_index": predicted[wrong].cpu().tolist(),
                "confidence": confidence[wrong].cpu().tolist(),
            }
        return None

    def on_validation_epoch_end(self) -> None:
        """记录四类各自的召回率，便于发现模型忽略少数类。"""
        recalls = self.val_recall.compute()
        for index, name in enumerate(self.class_names):
            self.log(f"val_recall_{name}", recalls[index], on_step=False, on_epoch=True)
        self.val_recall.reset()

    def configure_optimizers(self) -> torch.optim.Optimizer | dict:
        """交由 Lightning 按轮次推进学习率；旧实验仍可选 plateau。"""
        optimizer = torch.optim.AdamW(self.backbone.parameters(), lr=self.learning_rate,
                                      weight_decay=self.weight_decay)
        if self.lr_scheduler == "none":
            return optimizer
        if self.lr_scheduler == "cosine":
            warmup_epochs = min(COSINE_WARMUP_EPOCHS, max(0, self.schedule_epochs - 2))
            schedulers = []
            milestones = []
            if warmup_epochs:
                schedulers.append(torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=COSINE_WARMUP_START_FACTOR,
                    total_iters=warmup_epochs,
                ))
                milestones.append(warmup_epochs)
            schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.schedule_epochs - warmup_epochs,
                eta_min=self.learning_rate * COSINE_MIN_LR_RATIO,
            ))
            # 续训超过原计划轮数时维持最低学习率，不让余弦曲线反弹。
            schedulers.append(torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=COSINE_MIN_LR_RATIO, total_iters=2**31 - 1,
            ))
            milestones.append(self.schedule_epochs)
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=schedulers, milestones=milestones,
            )
            return {"optimizer": optimizer, "lr_scheduler": {
                "scheduler": scheduler, "interval": "epoch",
            }}
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=PLATEAU_FACTOR,
            patience=PLATEAU_PATIENCE, cooldown=PLATEAU_COOLDOWN,
            min_lr=self.learning_rate * PLATEAU_MIN_LR_RATIO,
        )
        return {"optimizer": optimizer, "lr_scheduler": {
            "scheduler": scheduler, "monitor": "val_macro_f1", "interval": "epoch",
        }}
