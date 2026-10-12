"""事后排除 CRF049 单核细胞作敏感性分析；不覆盖正式测试报告。"""

import csv
import json
import tempfile
from pathlib import Path

from leukocyte_analysis.data.bbbc045_config import CLASS_NAMES
from leukocyte_analysis.evaluation.metrics import classification_metrics
from leukocyte_analysis.evaluation.plots import save_confusion_matrices


PROJECT = Path(__file__).resolve().parents[1]
EVALUATION = (
    PROJECT
    / "runs"
    / "resnet18_four_imagenet_flips_20261011_143841"
    / "test_evaluation"
)
PREDICTIONS = EVALUATION / "test_predictions.csv"
OUTPUT = EVALUATION / "posthoc_excluding_crf049_monocyte"


def metrics(rows):
    """用 scikit-learn 对指定子集重算指标，不重新推理模型。"""
    y_true = [int(row["true_index"]) for row in rows]
    y_pred = [int(row["predicted_index"]) for row in rows]
    result = classification_metrics(y_true, y_pred, CLASS_NAMES)
    return {
        "samples": result["samples"],
        "errors": sum(true != pred for true, pred in zip(y_true, y_pred)),
        "accuracy": result["accuracy"],
        "macro_f1": result["macro_f1"],
        "classification_report": result["classification_report"],
    }


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    """生成单独的事后报告，保留原始测试报告不变。"""
    if OUTPUT.exists():
        raise FileExistsError(f"Report already exists; refusing to overwrite: {OUTPUT}")

    with PREDICTIONS.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError("Prediction CSV has no header")
        predictions = list(reader)

    excluded = [
        row for row in predictions
        if row["donor"] == "CRF049" and row["true_label"] == "monocyte"
    ]
    retained = [
        row for row in predictions
        if not (row["donor"] == "CRF049" and row["true_label"] == "monocyte")
    ]
    if len(predictions) != 15290 or len(excluded) != 376:
        raise ValueError(
            f"Unexpected input counts: all={len(predictions)}, excluded={len(excluded)}"
        )

    summary = {
        "analysis": "post-hoc sensitivity analysis, not an unbiased final test score",
        "source_predictions": str(PREDICTIONS),
        "exclusion_rule": {
            "donor": "CRF049",
            "true_label": "monocyte",
            "note": "All matching samples excluded, not only model errors",
        },
        "original": metrics(predictions),
        "excluded_subset": metrics(excluded),
        "retained_subset": metrics(retained),
    }

    with tempfile.TemporaryDirectory(prefix=".sensitivity-", dir=EVALUATION) as temporary:
        staged = Path(temporary) / "report"
        staged.mkdir()
        write_csv(staged / "excluded_predictions.csv", excluded, fieldnames)
        write_csv(staged / "retained_predictions.csv", retained, fieldnames)
        with (staged / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        y_true = [int(row["true_index"]) for row in retained]
        y_pred = [int(row["predicted_index"]) for row in retained]
        save_confusion_matrices(y_true, y_pred, staged, CLASS_NAMES)
        staged.rename(OUTPUT)

    for name in ("original", "excluded_subset", "retained_subset"):
        result = summary[name]
        print(
            name,
            f"samples={result['samples']}",
            f"errors={result['errors']}",
            f"accuracy={result['accuracy']:.4f}",
            f"macro_f1={result['macro_f1']:.4f}",
        )
    print(f"Report: {OUTPUT}")


if __name__ == "__main__":
    main()
