"""固定单次训练权重，诊断 CRF049 单核细胞对亮度/对比度变化的敏感性。"""

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
CHECKPOINT = ROOT / "runs/resnet18_four_imagenet_flips_20261011_143841/checkpoints/best.ckpt"
MANIFEST = ROOT / "data/processed/bbbc045_split.csv"
TILES = ROOT / "data/processed/bbbc045_tiles.npy"
STATS = ROOT / "data/processed/bbbc045_train_stats.json"
REPORT = ROOT / "reports/crf049_monocyte_photometric_diagnostic.json"
MONOCYTE = CLASS_NAMES.index("monocyte")


def select_indices() -> dict[str, list[int]]:
    """保留原清单行号，训练单核只作亮度参照，绝不送入推理统计。"""
    selected = {"train_monocyte": [], "CRF049_batch1": [],
                "CRF049_batch2": [], "CRF066": []}
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if row["label"] != "monocyte":
                continue
            if row["split"] == "train":
                selected["train_monocyte"].append(index)
            elif row["split"] == "test" and row["donor"] == "CRF049":
                batch = row["file"].replace("\\", "/").split("/")[2]
                if batch not in {"batch1", "batch2"}:
                    raise ValueError(f"未知 CRF049 单核批次：{batch}")
                selected[f"CRF049_{batch}"].append(index)
            elif row["split"] == "test" and row["donor"] == "CRF066":
                selected["CRF066"].append(index)
    if (len(selected["CRF049_batch1"]), len(selected["CRF049_batch2"]),
            len(selected["CRF066"])) != (320, 56, 212):
        raise ValueError("测试集单核细胞数量与已有报告不一致")
    return selected


@torch.inference_mode()
def summarize(model: torch.nn.Module, raw: np.ndarray, mean: float, std: float,
              device: torch.device) -> dict:
    """沿用原训练集全局标准化和类别顺序，统计固定权重预测。"""
    guesses = []
    mono_prob = []
    for start in range(0, len(raw), 128):
        normalized = ((raw[start:start + 128] - mean) / std).astype(np.float32)
        probabilities = model(torch.from_numpy(normalized[:, None]).to(device))
        probabilities = probabilities.softmax(dim=1).cpu().numpy()
        guesses.extend(probabilities.argmax(axis=1).tolist())
        mono_prob.extend(probabilities[:, MONOCYTE].tolist())
    counts = Counter(CLASS_NAMES[index] for index in guesses)
    correct = counts["monocyte"]
    return {
        "n": len(raw), "predicted_as_monocyte": correct,
        "recall": correct / len(raw),
        "predicted_counts": {name: counts[name] for name in CLASS_NAMES},
        "median_monocyte_probability": float(np.median(mono_prob)),
    }


def variants(raw: np.ndarray, reference_mean: float, reference_std: float) -> dict:
    """在看新预测前固定多档变换；类别均值匹配是事后诊断，不可部署。"""
    source_mean, source_std = float(raw.mean()), float(raw.std())
    ratio = reference_std / source_std
    return {
        "original": raw,
        "match_train_monocyte_mean": raw + reference_mean - source_mean,
        "match_train_monocyte_mean_ratio": raw * (reference_mean / source_mean),
        "match_train_monocyte_contrast_only": (raw - source_mean) * ratio + source_mean,
        "match_train_monocyte_mean_std": (raw - source_mean) * ratio + reference_mean,
        "brightness_x0.8": raw * 0.8,
        "brightness_x0.9": raw * 0.9,
        "brightness_x1.1": raw * 1.1,
        "brightness_x1.2": raw * 1.2,
        "brightness_x1.25": raw * 1.25,
    }


def main() -> None:
    """固定原测试权重与校验数字，比较各批单核图块的亮度扰动。"""
    if REPORT.exists():
        raise FileExistsError(f"为避免覆盖先前诊断，报告已存在：{REPORT}")
    selected = select_indices()
    tiles = np.load(TILES, mmap_mode="r", allow_pickle=False)
    train_mono = np.asarray(tiles[selected["train_monocyte"]], dtype=np.float32)
    reference_mean, reference_std = float(train_mono.mean()), float(train_mono.std())
    with STATS.open(encoding="utf-8") as handle:
        stats = json.load(handle)

    backbone = build_resnet18(len(CLASS_NAMES), initialization="random")
    model = load_classifier(CHECKPOINT, backbone)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    groups = {}
    for name in ("CRF049_batch1", "CRF049_batch2", "CRF066"):
        raw = np.asarray(tiles[selected[name]], dtype=np.float32)
        results = {
            variant: summarize(model, corrected, stats["mean"], stats["std"], device)
            for variant, corrected in variants(raw, reference_mean, reference_std).items()
        }
        groups[name] = {"raw_mean": float(raw.mean()), "raw_std": float(raw.std()),
                        "results": results}
    baseline = [groups[name]["results"]["original"]["predicted_as_monocyte"]
                for name in ("CRF049_batch1", "CRF049_batch2", "CRF066")]
    if baseline != [0, 8, 208]:
        raise RuntimeError(f"原图预测未重现正式测试报告：{baseline}")

    report = {
        "purpose": "固定已查看过的测试集权重，仅作机制诊断；不是独立测试成绩",
        "checkpoint": str(CHECKPOINT),
        "train_monocyte_reference": {"n": len(train_mono), "mean": reference_mean,
                                     "std": reference_std},
        "groups": groups,
        "interpretation_limit": "单核类别均值匹配使用了原始测试标签和测试图像统计；不能用于模型选择、部署或声称独立泛化。",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, group in groups.items():
        print(name, "原始均值", group["raw_mean"])
        for variant, result in group["results"].items():
            print(f"  {variant}: {result['predicted_as_monocyte']}/{result['n']}",
                  result["predicted_counts"])
    print("诊断报告：", REPORT)


if __name__ == "__main__":
    main()
