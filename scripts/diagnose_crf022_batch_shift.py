"""固定交叉验证权重，诊断 CRF022/T/batch2 对亮度与对比度的敏感性。"""

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from leukocyte_analysis.data.bbbc045_config import CLASS_NAMES
from leukocyte_analysis.evaluation.runner import load_classifier
from leukocyte_analysis.models.resnet import build_resnet18


ROOT = Path(__file__).resolve().parents[1]
FOLD = ROOT / "runs/cv_resnet18_imagenet_flips_20261011_213201/fold_01"
TILES = ROOT / "data/processed/bbbc045_tiles.npy"
REPORT = ROOT / "reports/crf022_t_batch2_photometric_diagnostic.json"


def batch_indices(split_csv: Path) -> dict[str, list[int]]:
    """按原始目录筛出第 1 折验证集中的 CRF022/T 两批。"""
    indices = {"batch1": [], "batch2": []}
    with split_csv.open(newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            parts = row["file"].replace("\\", "/").split("/")
            if (row["donor"], row["label"], row["split"]) != ("CRF022", "T", "val"):
                continue
            if parts[:2] == ["CRF022", "T"] and parts[2] in indices:
                indices[parts[2]].append(index)
    if (len(indices["batch1"]), len(indices["batch2"])) != (901, 915):
        raise ValueError("CRF022/T 两批的行号或样本数与原交叉验证不符")
    return indices


@torch.inference_mode()
def summarize(model: torch.nn.Module, raw: np.ndarray, mean: float, std: float,
              device: torch.device) -> dict:
    """沿用训练集全局标准化，只汇总 T→淋巴细胞的固定权重预测。"""
    predicted = []
    true_prob = []
    for start in range(0, len(raw), 128):
        normalized = ((raw[start:start + 128] - mean) / std).astype(np.float32)
        images = torch.from_numpy(normalized[:, None]).to(device)
        probabilities = model(images).softmax(dim=1).cpu().numpy()
        predicted.extend(probabilities.argmax(axis=1).tolist())
        true_prob.extend(probabilities[:, 0].tolist())
    correct = sum(index == 0 for index in predicted)
    counts = Counter(CLASS_NAMES[index] for index in predicted)
    return {
        "n": len(raw), "correct_as_lymphocyte": correct,
        "recall": correct / len(raw),
        "predicted_counts": {name: counts[name] for name in CLASS_NAMES},
        "median_lymphocyte_probability": float(np.median(true_prob)),
    }


def main() -> None:
    """使用 batch1 的亮度/对比度作参照，不修改磁盘上的图块。"""
    if REPORT.exists():
        raise FileExistsError(f"为避免覆盖先前诊断，报告已存在：{REPORT}")
    indices = batch_indices(FOLD / "data/split.csv")
    tiles = np.load(TILES, mmap_mode="r", allow_pickle=False)
    batch1 = np.asarray(tiles[indices["batch1"]], dtype=np.float32)
    batch2 = np.asarray(tiles[indices["batch2"]], dtype=np.float32)
    mu1, mu2 = float(batch1.mean()), float(batch2.mean())
    sd1, sd2 = float(batch1.std()), float(batch2.std())
    delta, contrast_ratio = mu1 - mu2, sd1 / sd2
    with (FOLD / "data/train_stats.json").open(encoding="utf-8") as handle:
        stats = json.load(handle)

    backbone = build_resnet18(len(CLASS_NAMES), initialization="random")
    model = load_classifier(FOLD / "training/checkpoints/best.ckpt", backbone)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # 档位在查看本次预测前固定；均值/标准差匹配仅用于诊断，不是正式预处理。
    variants = {
        "original": batch2,
        "add_half_mean_gap": batch2 + delta / 2,
        "add_full_mean_gap": batch2 + delta,
        "multiply_mean_ratio": batch2 * (mu1 / mu2),
        "contrast_only": (batch2 - mu2) * contrast_ratio + mu2,
        "match_mean_and_std": (batch2 - mu2) * contrast_ratio + mu1,
    }
    control = summarize(model, batch1, stats["mean"], stats["std"], device)
    results = {
        name: summarize(model, corrected, stats["mean"], stats["std"], device)
        for name, corrected in variants.items()
    }
    if control["correct_as_lymphocyte"] != 890 or results["original"]["correct_as_lymphocyte"] != 0:
        raise RuntimeError("原图预测未重现交叉验证结果；停止，不发布诊断报告")

    report = {
        "purpose": "固定交叉验证第 1 折权重的验证集诊断；不是独立测试成绩",
        "checkpoint": str(FOLD / "training/checkpoints/best.ckpt"),
        "source": "CRF022/T；B 与 T 在四分类中均映射为 lymphocyte",
        "raw_statistics": {"batch1_mean": mu1, "batch2_mean": mu2,
                           "batch1_std": sd1, "batch2_std": sd2,
                           "mean_gap": delta, "contrast_ratio": contrast_ratio},
        "batch1_unchanged_control": control,
        "batch2_variants": results,
        "interpretation_limit": "校正参数参考同一验证受试者；只能用于机制排查，不能用于声称独立泛化或更正标签。",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("原图对照：", control["correct_as_lymphocyte"], "/", control["n"])
    for name, result in results.items():
        print(f"{name}: {result['correct_as_lymphocyte']}/{result['n']} ",
              result["predicted_counts"])
    print("诊断报告：", REPORT)


if __name__ == "__main__":
    main()
