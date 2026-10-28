"""把模型错分与原始 TIFF 网格坐标关联，供人工复核标签。"""

from ..data.bbbc045_config import model_label


ERROR_FIELDS = (
    "split", "epoch", "packed_index", "file", "donor", "row", "col",
    "source_label", "true_label", "predicted_label", "confidence",
)
FREQUENCY_FIELDS = (
    "packed_index", "file", "donor", "row", "col", "source_label",
    "true_label", "wrong_epochs", "evaluated_epochs", "wrong_rate",
    "epochs", "predicted_labels",
)


def misclassified_rows(dataset, predictions: list[dict], split: str,
                       epoch: int | None = None) -> list[dict]:
    """仅保留错分，并从数据集清单补回可追溯的来源位置。"""
    rows = []
    for prediction in predictions:
        packed_index = int(prediction["packed_index"])
        true_index = int(prediction["true_index"])
        predicted_index = int(prediction["predicted_index"])
        source = dataset.source_by_index[packed_index]
        true_label = dataset.class_names[true_index]
        if model_label(source["source_label"]) != true_label:
            raise ValueError(f"第 {packed_index} 个图块的原始标签与预测记录不一致")
        if prediction.get("donor", source["donor"]) != source["donor"]:
            raise ValueError(f"第 {packed_index} 个图块的受试者不一致")
        if prediction.get("true_label", true_label) != true_label:
            raise ValueError(f"第 {packed_index} 个图块的模型标签不一致")
        if true_index == predicted_index:
            continue

        rows.append({
            "split": split,
            "epoch": "" if epoch is None else epoch,
            "packed_index": packed_index,
            "file": source["file"],
            "donor": source["donor"],
            "row": source["row"],
            "col": source["col"],
            "source_label": source["source_label"],
            "true_label": true_label,
            "predicted_label": dataset.class_names[predicted_index],
            "confidence": float(prediction["confidence"]),
        })
    return rows
