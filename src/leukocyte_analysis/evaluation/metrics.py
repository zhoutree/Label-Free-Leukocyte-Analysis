"""共享分类指标；由 scikit-learn 统一计算。"""

from collections.abc import Sequence

from sklearn.metrics import accuracy_score, classification_report


def classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str]
) -> dict:
    """返回准确率、macro-F1 和完整逐类报告。"""
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("真实标签与预测标签必须非空且长度一致")
    report = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=list(class_names), output_dict=True, zero_division=0,
    )
    return {
        "samples": len(y_true),
        "class_names": list(class_names),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "classification_report": report,
    }
