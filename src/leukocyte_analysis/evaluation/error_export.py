"""从已有预测或验证检查点导出错分，供命令行及其他调用方复用。"""

import csv
from pathlib import Path

import torch

from ..data.bbbc045_config import CLASS_NAMES
from ..data.dataset import LeukocyteDataset
from ..models.resnet import build_resnet18
from ..reporting import write_csv
from .error_records import ERROR_FIELDS, misclassified_rows
from .inference import check_checkpoint_head, load_classifier, predict


def export_errors(split_csv: Path, tiles_npy: Path, stats_json: Path, split: str,
                  output_csv: Path, *, predictions_csv: Path | None = None,
                  checkpoint: Path | None = None, batch_size: int = 128) -> int:
    """返回导出的错分数量；保护已有文件，测试集只读取既有预测。

    核对完整预测的图块顺序及来源标签后写出错分。不修改原图、
    清单或权重，也不允许绕过 CLI 直接对测试集重复推理。
    """
    if split not in {"val", "test"} or batch_size < 1:
        raise ValueError("只支持 val／test，且 batch-size 必须大于 0")
    if (predictions_csv is None) == (checkpoint is None):
        raise ValueError("预测 CSV 与检查点必须且只能指定一个")
    if checkpoint is not None and split != "val":
        raise ValueError("测试集只接受既有预测 CSV，避免重复运行测试推理")
    if output_csv.exists():
        raise FileExistsError(f"错分记录已存在：{output_csv}")
    dataset = LeukocyteDataset(split_csv, tiles_npy, stats_json, split)
    if predictions_csv is not None:
        with predictions_csv.open("r", newline="", encoding="utf-8") as handle:
            predictions = list(csv.DictReader(handle))
    else:
        check_checkpoint_head(checkpoint)
        backbone = build_resnet18(len(CLASS_NAMES), initialization="random")
        model = load_classifier(checkpoint, backbone)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        predictions = predict(model, dataset, batch_size, device)
    expected_indices = [index for index, _label, _donor in dataset.samples]
    actual_indices = [int(row["packed_index"]) for row in predictions]
    if actual_indices != expected_indices:
        raise ValueError("预测文件与当前数据组的 CSV 行号或顺序不一致")
    rows = misclassified_rows(dataset, predictions, split)
    write_csv(output_csv, rows, ERROR_FIELDS)
    return len(rows)
