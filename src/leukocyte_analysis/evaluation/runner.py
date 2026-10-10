"""加载最佳检查点，对保留测试集推理并输出可复核报告。"""

import csv
import json
import tempfile
from pathlib import Path

import lightning as L
import torch
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
from torch.utils.data import DataLoader

from ..data.bbbc045_config import CLASS_NAMES
from ..data.dataset import LeukocyteDataset
from ..models.resnet import build_resnet18
from ..training.module import LeukocyteClassifier
from .plots import save_confusion_matrices, save_error_examples


def load_classifier(checkpoint: Path, backbone: nn.Module) -> LeukocyteClassifier:
    """从本项目生成的 Lightning 检查点恢复网络，无须再下载预训练权重。"""
    return LeukocyteClassifier.load_from_checkpoint(
        str(checkpoint), backbone=backbone,
        class_weights=torch.ones(len(CLASS_NAMES)), map_location="cpu",
        weights_only=False,  # 仅加载自己训练的可信检查点。
    )


def predict(model: LeukocyteClassifier, dataset: LeukocyteDataset, batch_size: int,
            device: torch.device) -> list[dict]:
    """由 Lightning 接管顺序推理，再将输出关联到清单索引。"""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    trainer = L.Trainer(
        accelerator="gpu" if device.type == "cuda" else "cpu", devices=1,
        logger=False, enable_checkpointing=False, enable_model_summary=False,
    )
    # predict_step 已把三个输出移至 CPU；测试集结果可安全保存在内存中。
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
                "true_index": target, "true_label": CLASS_NAMES[target],
                "predicted_index": guess, "predicted_label": CLASS_NAMES[guess],
                "confidence": score,
            })
            offset += 1
    if offset != len(dataset):
        raise ValueError("推理结果数量与测试集不一致")
    return records


def save_report(dataset: LeukocyteDataset, predictions: list[dict], output_dir: Path) -> dict:
    """先在临时目录生成完整报告，成功后一次性发布。"""
    if not predictions:
        raise ValueError("没有可评估的细胞")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise FileExistsError(f"报告目录已存在，避免覆盖或重复评估：{output_dir}")
    with tempfile.TemporaryDirectory(prefix=".evaluation-", dir=output_dir.parent) as temporary:
        staged_dir = Path(temporary) / "report"
        staged_dir.mkdir()
        summary = _write_report(dataset, predictions, staged_dir)
        # 同一磁盘内重命名；中途异常不会留下半成品正式报告。
        staged_dir.rename(output_dir)
    return summary


def _write_report(dataset: LeukocyteDataset, predictions: list[dict], output_dir: Path) -> dict:
    """以 scikit-learn 计算指标，并写入暂存目录。"""
    y_true = [row["true_index"] for row in predictions]
    y_pred = [row["predicted_index"] for row in predictions]
    options = dict(labels=list(range(len(CLASS_NAMES))), target_names=list(CLASS_NAMES),
                   zero_division=0)
    report = classification_report(y_true, y_pred, output_dict=True, **options)
    summary = {
        "samples": len(predictions),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "classification_report": report,
    }
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    (output_dir / "test_classification_report.txt").write_text(
        classification_report(y_true, y_pred, output_dict=False, **options), encoding="utf-8"
    )
    with (output_dir / "test_per_class_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("class", "precision", "recall", "f1-score", "support"))
        writer.writeheader()
        for name in CLASS_NAMES:
            writer.writerow({"class": name, **report[name]})
    with (output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    save_confusion_matrices(y_true, y_pred, output_dir)
    save_error_examples(dataset, predictions, output_dir / "misclassified_examples.png")
    return summary


def evaluate_checkpoint(checkpoint: Path, split_csv: Path, tiles_npy: Path,
                        stats_json: Path, output_dir: Path, batch_size: int) -> dict:
    """只读取 test 划分；测试集不参与权重或超参数选择。"""
    for path in (checkpoint, split_csv, tiles_npy, stats_json):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists():
        raise FileExistsError(f"报告目录已存在，避免覆盖或重复评估：{output_dir}")
    dataset = LeukocyteDataset(split_csv, tiles_npy, stats_json, "test")
    backbone = build_resnet18(len(CLASS_NAMES), initialization="random")
    model = load_classifier(checkpoint, backbone)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = predict(model, dataset, batch_size, device)
    return save_report(dataset, predictions, output_dir)
