"""使用 Matplotlib 生成评估与亮度诊断图。"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 命令行和无显示器环境也可以保存图片。
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def save_brightness_comparison(overall: list[dict], classes: list[dict],
                               output_dir: Path, class_names: tuple[str, ...]) -> None:
    """按相同亮度倍率绘制两种训练方式的总体指标与逐类召回率。"""
    runs = list(dict.fromkeys(row["run"] for row in overall))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for run in runs:
        rows = sorted((row for row in overall if row["run"] == run),
                      key=lambda row: row["brightness_factor"])
        for axis, metric in zip(axes, ("accuracy", "macro_f1")):
            axis.plot([row["brightness_factor"] for row in rows],
                      [row[metric] for row in rows], marker="o", label=run)
    for axis, metric in zip(axes, ("accuracy", "macro_f1")):
        axis.set(xlabel="Brightness multiplier", ylabel=metric, ylim=(0, 1.02))
        axis.grid(alpha=0.3)
    axes[0].legend()
    fig.savefig(output_dir / "overall.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for axis, name in zip(axes.flat, class_names):
        for run in runs:
            rows = sorted((row for row in classes if row["run"] == run
                           and row["class"] == name),
                          key=lambda row: row["brightness_factor"])
            axis.plot([row["brightness_factor"] for row in rows],
                      [row["recall"] for row in rows], marker="o", label=run)
        axis.set(title=name, xlabel="Brightness multiplier", ylabel="Recall", ylim=(0, 1.02))
        axis.grid(alpha=0.3)
    axes.flat[0].legend()
    fig.savefig(output_dir / "per_class_recall.png", dpi=160)
    plt.close(fig)


def save_confusion_matrices(y_true: list[int], y_pred: list[int], output_dir: Path,
                            class_names: tuple[str, ...]) -> None:
    """分别保存计数和按真实类别归一化的混淆矩阵。"""
    for normalize, suffix, title in (
        (None, "counts", "Confusion matrix: counts"),
        ("true", "row_normalized", "Confusion matrix: recall by true class"),
    ):
        fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, labels=range(len(class_names)),
            display_labels=class_names, normalize=normalize,
            values_format="d" if normalize is None else ".2f", ax=ax,
            colorbar=True, cmap="Blues",
        )
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=35)
        fig.savefig(output_dir / f"confusion_matrix_{suffix}.png", dpi=180)
        plt.close(fig)


def save_error_examples(dataset, predictions: list[dict], output_path: Path,
                        class_names: tuple[str, ...]) -> None:
    """每个真实类别最多选三张高置信度错分图；拉伸只用于展示。"""
    errors = [row for row in predictions if row["true_index"] != row["predicted_index"]]
    selected = []
    for label in range(len(class_names)):
        candidates = (row for row in errors if row["true_index"] == label)
        selected.extend(sorted(candidates, key=lambda row: -row["confidence"])[:3])

    if not selected:
        fig, ax = plt.subplots(figsize=(7, 2))
        ax.text(0.5, 0.5, "No misclassified cells", ha="center", va="center")
        ax.axis("off")
    else:
        columns = 3
        rows = (len(selected) + columns - 1) // columns
        fig, axes = plt.subplots(rows, columns, figsize=(columns * 3, rows * 3.2), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        tiles = dataset._get_tiles()
        for ax, record in zip(axes.flat, selected):
            tile = np.asarray(tiles[record["packed_index"]])
            low, high = np.percentile(tile, (1, 99))
            # 逐图拉伸仅为让人看清；模型输入与分类结果均未改变。
            ax.imshow(tile, cmap="gray", vmin=low, vmax=high if high > low else low + 1)
            ax.set_title(
                f"{record['donor']} #{record['packed_index']}\n"
                f"{record['true_label']} → {record['predicted_label']} ({record['confidence']:.2f})",
                fontsize=9,
            )
            ax.axis("off")
        fig.suptitle("Highest-confidence errors per true class (display: 1–99% stretch)")
        fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
