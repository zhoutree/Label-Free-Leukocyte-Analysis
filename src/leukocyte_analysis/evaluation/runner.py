"""为保留测试集生成可复核报告；通用推理实现见 inference。"""

import json
from pathlib import Path

import torch
from sklearn.metrics import classification_report

from ..data.bbbc045_config import CLASS_NAMES, TEST_DONORS
from ..data.dataset import LeukocyteDataset
from ..data.split import read_manifest, validate_fixed_splits
from ..models.resnet import build_resnet18
from ..reporting import new_report, write_csv
from .error_records import ERROR_FIELDS, misclassified_rows
from .inference import check_checkpoint_head, load_classifier, predict
from .metrics import classification_metrics
from .plots import save_confusion_matrices, save_error_examples


def save_report(dataset: LeukocyteDataset, predictions: list[dict], output_dir: Path) -> dict:
    """先在临时目录生成完整报告，成功后一次性发布。"""
    if not predictions:
        raise ValueError("没有可评估的细胞")
    with new_report(output_dir, prefix=".evaluation-") as staged_dir:
        summary = _write_report(dataset, predictions, staged_dir)
    return summary


def _write_report(dataset: LeukocyteDataset, predictions: list[dict], output_dir: Path) -> dict:
    """以 scikit-learn 计算指标，并写入暂存目录。"""
    class_names = dataset.class_names
    y_true = [row["true_index"] for row in predictions]
    y_pred = [row["predicted_index"] for row in predictions]
    options = dict(labels=list(range(len(class_names))), target_names=list(class_names),
                   zero_division=0)
    summary = classification_metrics(y_true, y_pred, class_names)
    report = summary["classification_report"]
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    (output_dir / "test_classification_report.txt").write_text(
        classification_report(y_true, y_pred, output_dict=False, **options), encoding="utf-8"
    )
    class_fields = ("class", "precision", "recall", "f1-score", "support")
    class_rows = [{"class": name, **report[name]} for name in class_names]
    write_csv(output_dir / "test_per_class_metrics.csv", class_rows, class_fields)
    write_csv(output_dir / "test_predictions.csv", predictions, tuple(predictions[0]))
    # 完整预测用于重算指标；错分表额外保留原始文件和网格坐标。
    errors = misclassified_rows(dataset, predictions, "test")
    write_csv(output_dir / "test_misclassified.csv", errors, ERROR_FIELDS)
    save_confusion_matrices(y_true, y_pred, output_dir, class_names)
    save_error_examples(dataset, predictions, output_dir / "misclassified_examples.png", class_names)
    return summary


def _validate_heldout_test_split(split_csv: Path) -> None:
    """正式测试只允许固定划分，且须包含两个原保留受试者。"""
    _fields, records = read_manifest(split_csv)
    validate_fixed_splits(records)
    donors = {row["donor"] for row in records if row["split"] == "test"}
    if donors != TEST_DONORS:
        raise ValueError(f"正式测试必须包含原保留受试者 {', '.join(sorted(TEST_DONORS))}")


def evaluate_checkpoint(checkpoint: Path, split_csv: Path, tiles_npy: Path,
                        stats_json: Path, output_dir: Path, batch_size: int) -> dict:
    """只对 ``test`` 划分推理，并在新目录发布可复核报告。

    使用传入的训练统计量标准化图块；测试集不参与权重或超参数选择。
    返回总体准确率、macro-F1 及逐类报告，拒绝覆盖已有结果。
    """
    for path in (checkpoint, split_csv, tiles_npy, stats_json):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists():
        raise FileExistsError(f"报告目录已存在，避免覆盖或重复评估：{output_dir}")
    _validate_heldout_test_split(split_csv)
    check_checkpoint_head(checkpoint)
    dataset = LeukocyteDataset(split_csv, tiles_npy, stats_json, "test")
    backbone = build_resnet18(len(CLASS_NAMES), initialization="random")
    model = load_classifier(checkpoint, backbone)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = predict(model, dataset, batch_size, device)
    return save_report(dataset, predictions, output_dir)
