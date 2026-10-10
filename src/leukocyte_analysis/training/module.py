"""使用 Lightning 管理优化步骤，使用 TorchMetrics 汇总整轮指标。"""

import lightning as L
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassF1Score, MulticlassRecall

from ..data.bbbc045_config import CLASS_NAMES


class LeukocyteClassifier(L.LightningModule):
    """保留项目特有的加权损失；交由 Lightning 管理循环与设备。"""

    def __init__(
        self,
        backbone: nn.Module,
        class_weights: torch.Tensor,
        learning_rate: float,
        weight_decay: float,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=("backbone", "class_weights"))
        self.backbone = backbone
        self.register_buffer("class_weights", class_weights.detach().clone())
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.train_loss = MeanMetric()
        self.train_weighted_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_f1 = MulticlassF1Score(num_classes=len(CLASS_NAMES), average="macro")
        self.val_recall = MulticlassRecall(num_classes=len(CLASS_NAMES), average=None)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """返回每类的原始分数，不在模型内做 softmax。"""
        return self.backbone(images)

    def predict_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """供 Trainer.predict 使用；返回标签、预测类别和置信度。"""
        images, targets = batch
        confidence, predicted = self(images).softmax(dim=1).max(dim=1)
        # 及时转回 CPU，避免 Lightning 汇集全部预测时占用显存。
        return targets.cpu(), predicted.cpu(), confidence.cpu()

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """加权 CE 更新参数；未加权 CE 用于和验证曲线比较。"""
        images, targets = batch
        logits = self(images)
        objective = F.cross_entropy(logits, targets, weight=self.class_weights)
        per_cell_loss = F.cross_entropy(logits.detach(), targets, reduction="none")

        # MeanMetric 按细胞而非按批次汇总，避免最后一个小批次造成偏差。
        self.train_loss.update(per_cell_loss)
        self.train_weighted_loss.update(per_cell_loss, weight=self.class_weights[targets])
        self.log("train_loss", self.train_loss, on_step=False, on_epoch=True)
        self.log("train_weighted_loss", self.train_weighted_loss, on_step=False, on_epoch=True)
        self.log("step_weighted_loss", objective.detach(), on_step=True, on_epoch=False,
                 prog_bar=True, logger=False)
        return objective

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """验证集保持原始分布；指标只在整轮结束后汇总。"""
        images, targets = batch
        logits = self(images)
        self.val_loss.update(F.cross_entropy(logits, targets, reduction="none"))
        self.val_f1.update(logits, targets)
        self.val_recall.update(logits, targets)
        self.log("val_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_macro_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """记录五类各自的召回率，便于发现模型忽略少数类。"""
        recalls = self.val_recall.compute()
        for index, name in enumerate(CLASS_NAMES):
            self.log(f"val_recall_{name}", recalls[index], on_step=False, on_epoch=True)
        self.val_recall.reset()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """优化器由 Lightning 在适当设备上创建和运行。"""
        return torch.optim.AdamW(self.backbone.parameters(), lr=self.learning_rate,
                                 weight_decay=self.weight_decay)
