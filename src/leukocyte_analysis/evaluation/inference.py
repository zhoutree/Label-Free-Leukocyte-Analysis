"""共享四分类检查点加载与按清单索引的顺序推理。"""

from pathlib import Path

import lightning as L
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..data.bbbc045_config import CLASS_NAMES
from ..data.dataset import LeukocyteDataset
from ..training.module import LeukocyteClassifier
from ..training.progress import CompactRichProgressBar, prepare_console_output


def check_checkpoint_head(checkpoint: Path) -> None:
    """在推理前拒绝旧五分类检查点，避免标签含义与新模型错配。"""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    head = saved["state_dict"].get("backbone.fc.weight")
    if head is None or head.ndim != 2 or head.shape[0] != len(CLASS_NAMES):
        raise ValueError("只支持新四分类 ResNet 检查点；旧五分类权重不能用于四分类评估")


def load_classifier(checkpoint: Path, backbone: nn.Module) -> LeukocyteClassifier:
    """从本项目生成的 Lightning 检查点恢复网络，无须再下载预训练权重。"""
    return LeukocyteClassifier.load_from_checkpoint(
        str(checkpoint), backbone=backbone,
        class_weights=torch.ones(len(CLASS_NAMES)), map_location="cpu",
        weights_only=False,  # 仅加载自己训练的可信检查点。
    )


def predict(model: LeukocyteClassifier, dataset: LeukocyteDataset, batch_size: int,
            device: torch.device) -> list[dict]:
    """用 Lightning 顺序推理并返回带清单索引的逐图块预测。

    ``dataset.samples`` 的顺序必须与 DataLoader 输出一致；函数逐项
    核对真实标签和结果数量，返回 CPU 上的类别与置信度记录。
    """
    if tuple(model.class_names) != tuple(dataset.class_names):
        raise ValueError("模型与数据集的类别顺序不一致")
    prepare_console_output()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    trainer = L.Trainer(
        accelerator="gpu" if device.type == "cuda" else "cpu", devices=1,
        logger=False, enable_checkpointing=False, enable_model_summary=False,
        callbacks=[CompactRichProgressBar(refresh_rate=5)],
    )
    # predict_step 已把三个输出移至 CPU；验证或测试预测可安全保存在内存中。
    batches = trainer.predict(model, dataloaders=loader, return_predictions=True)
    if batches is None:
        raise RuntimeError("Lightning 没有返回预测结果")
    records = []
    offset = 0
    for targets, predicted, confidence in batches:
        for target, guess, score in zip(targets.tolist(), predicted.tolist(),
                                        confidence.tolist()):
            packed_index, label, donor = dataset.samples[offset]
            if target != label:
                raise ValueError("DataLoader 顺序与数据清单不一致")
            records.append({
                "packed_index": packed_index, "donor": donor,
                "true_index": target, "true_label": dataset.class_names[target],
                "predicted_index": guess, "predicted_label": dataset.class_names[guess],
                "confidence": score,
            })
            offset += 1
    if offset != len(dataset):
        raise ValueError("推理结果数量与数据集不一致")
    return records
