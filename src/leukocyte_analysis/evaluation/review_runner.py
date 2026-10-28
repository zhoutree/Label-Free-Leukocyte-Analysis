"""编排特征参考拟合和原始图块复核；统计拟合与规则由独立模块实现。"""

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.bbbc045_config import CLASS_NAMES, TILE_SIZE
from ..data.dataset import LeukocyteDataset
from ..data.integrity import file_sha256
from ..data.quality import focus_measurements
from ..data.split import read_reference_manifest
from ..models.resnet import build_resnet18
from ..reporting import new_report, write_csv
from ..paths import store_run_path
from .cell_review import ReviewPolicy
from .inference import check_checkpoint_head, load_classifier
from .novelty import (ResNetFeatures, detector_state, fit_detector, known_threshold,
                      outlier_scores, restore_detector)
from .quality_reference import load_quality_threshold


def _load_network(checkpoint: Path, device: torch.device) -> ResNetFeatures:
    """复用四分类检查点校验及恢复，不下载预训练权重。"""
    check_checkpoint_head(checkpoint)
    classifier = load_classifier(checkpoint, build_resnet18(len(CLASS_NAMES),
                                                          initialization="random"))
    return ResNetFeatures(classifier.backbone).to(device).eval()


def _features(network, dataset, batch_size: int, device: torch.device) -> tuple:
    """顺序提取无增强图块特征，及时移回 CPU，避免占满显存。"""
    features, targets, predictions, confidences = [], [], [], []
    try:
        with torch.inference_mode():
            for images, labels in DataLoader(dataset, batch_size=batch_size, shuffle=False):
                logits, description = network(images.to(device))
                confidence, predicted = logits.softmax(1).max(1)
                features.append(description.cpu())
                targets.append(labels)
                predictions.append(predicted.cpu())
                confidences.append(confidence.cpu())
    finally:
        # 即使前向失败，也释放数据集持有的 Windows 文件映射。
        dataset._tiles = None
    return tuple(torch.cat(values) for values in (features, targets, predictions, confidences))


def fit_reference(checkpoint: Path, split_csv: Path, tiles_npy: Path, stats_json: Path,
                  output_dir: Path, reject_rate: float, batch_size: int = 128) -> dict:
    """训练侧拟合 RMD、完整验证侧选择已知误拒工作点；不评估未知检出率。"""
    if batch_size < 1:
        raise ValueError("batch-size 必须大于 0")
    known_threshold(np.array([0.0]), reject_rate)
    read_reference_manifest(split_csv)
    # 尽早检查输出冲突；报告中途失败也不会发布半成品。
    with new_report(output_dir) as stage:
        train = LeukocyteDataset(split_csv, tiles_npy, stats_json, "train")
        validation = LeukocyteDataset(split_csv, tiles_npy, stats_json, "val")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        network = _load_network(checkpoint, device)
        print(f"提取训练参考特征：{len(train)} 张；不更新网络参数。")
        train_features, labels, _, _ = _features(network, train, batch_size, device)
        detector = fit_detector(train_features, labels)
        dimension = train_features.shape[1]
        del train_features, labels
        print(f"提取完整验证特征：{len(validation)} 张；仅校准已知类误拒率。")
        features, targets, predicted, confidences = _features(
            network, validation, batch_size, device)
        scores = outlier_scores(detector, features)
        threshold = known_threshold(scores, reject_rate)
        rows = []
        tiles = validation._get_tiles()
        for offset, (index, label, donor) in enumerate(validation.samples):
            rows.append({"packed_index": index, "donor": donor, "true_index": label,
                         "predicted_index": int(predicted[offset]),
                         "confidence": float(confidences[offset]),
                         "ood_score": float(scores[offset]),
                         **focus_measurements(tiles[index])})
        metadata = {
            "format_version": 1, "method": "RMD", "pytorch_ood_version": "0.4.0",
            "class_names": list(CLASS_NAMES), "feature_dimension": dimension,
            "checkpoint_sha256": file_sha256(checkpoint),
            "mean": train.mean, "std": train.std,
            "ood_threshold": threshold, "known_reject_rate_target": reject_rate,
            "train_samples": len(train), "calibration_samples": len(validation),
            "known_reject_rate_observed": float(np.mean(scores > threshold)),
            "per_class": [{"class": name, "samples": int((targets == index).sum()),
                           "rejected": int(np.sum((targets.numpy() == index) &
                                                  (scores > threshold)))}
                          for index, name in enumerate(CLASS_NAMES)],
            "sources": {name: {"path": store_run_path(path, output_dir),
                               "sha256": file_sha256(path)} for name, path in
                        (("checkpoint", checkpoint), ("split_csv", split_csv),
                         ("tiles_npy", tiles_npy), ("stats_json", stats_json))},
            "limitations": "已知四类验证校准，非未知检出验证；验证集已用于模型选择。"
                           "清晰度评分未自动确定不可用阈值；跨仪器和病理异常能力未验证。",
        }
        torch.save({"metadata": metadata, "state": detector_state(detector)}, stage / "rmd.pt")
        write_csv(stage / "validation_scores.csv", rows, tuple(rows[0]))
        (stage / "summary.json").write_text(json.dumps(metadata, ensure_ascii=False,
                                                      indent=2, allow_nan=False), encoding="utf-8")
    return metadata


def load_reference(reference: Path, checkpoint: Path) -> tuple:
    """用安全张量模式恢复状态，禁止换权重后沿用旧特征分布和阈值。"""
    saved = torch.load(reference, map_location="cpu", weights_only=True)
    metadata = saved["metadata"]
    if (metadata["format_version"] != 1 or metadata["method"] != "RMD"
            or metadata["pytorch_ood_version"] != "0.4.0"
            or metadata["class_names"] != list(CLASS_NAMES)
            or metadata["checkpoint_sha256"] != file_sha256(checkpoint)):
        raise ValueError("特征参考与检查点、类别或格式版本不匹配")
    if (not np.isfinite(metadata["mean"]) or not np.isfinite(metadata["std"])
            or metadata["std"] <= 0):
        raise ValueError("特征参考的训练归一化参数无效")
    ReviewPolicy(metadata["ood_threshold"])
    return restore_detector(saved["state"], metadata["feature_dimension"]), metadata


def review_tiles(checkpoint: Path, reference: Path, tiles_npy: Path, output_dir: Path,
                 *, batch_size: int = 128, focus_min: float | None = None,
                 focus_review_min: float | None = None,
                 quality_reference: Path | None = None) -> dict:
    """复核无标签原始灰度包；不缩放、不拉伸，必须与训练输入单位一致。

    接受 N×55×55 float32 数组；标准化参数绑定在参考文件中。输出保留原始
    候选类别，unusable／uncertain 是决策状态，不是新增的训练类别。
    """
    if batch_size < 1:
        raise ValueError("batch-size 必须大于 0")
    if quality_reference is not None:
        if focus_min is not None:
            raise ValueError("质量参考与手动 focus-min 不能同时使用")
        focus_min = load_quality_threshold(quality_reference)
    detector, metadata = load_reference(reference, checkpoint)
    policy = ReviewPolicy(metadata["ood_threshold"], focus_min, focus_review_min)
    tiles = np.load(tiles_npy, mmap_mode="r", allow_pickle=False)
    try:
        if (tiles.ndim != 3 or tiles.shape[1:] != (TILE_SIZE, TILE_SIZE)
                or len(tiles) == 0 or tiles.dtype != np.float32):
            raise ValueError("输入须为非空 N×55×55 float32 原始灰度图块包")
        with new_report(output_dir) as stage:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            network = _load_network(checkpoint, device)
            rows = []
            with torch.inference_mode():
                for start in range(0, len(tiles), batch_size):
                    raw = np.array(tiles[start:start + batch_size], dtype=np.float32, copy=True)
                    qualities = [focus_measurements(tile) for tile in raw]
                    normalized = (raw - metadata["mean"]) / metadata["std"]
                    if not np.isfinite(normalized).all():
                        raise ValueError("图块标准化后出现非有限像素")
                    logits, features = network(torch.from_numpy(normalized[:, None]).to(device))
                    confidences, predicted = logits.softmax(1).max(1)
                    scores = outlier_scores(detector, features)
                    predicted = predicted.cpu().tolist()
                    confidences = confidences.cpu().tolist()
                    for offset, quality in enumerate(qualities):
                        guess, confidence = int(predicted[offset]), float(confidences[offset])
                        score = float(scores[offset])
                        rows.append({"packed_index": start + offset, "predicted_index": guess,
                                     "predicted_label": CLASS_NAMES[guess], "confidence": confidence,
                                     "ood_score": score, **quality,
                                     **policy.assess(guess, score, quality)})
            write_csv(stage / "predictions_with_review.csv", rows, tuple(rows[0]))
            write_csv(stage / "needs_review.csv", [row for row in rows if
                      row["decision"] != "accepted"], tuple(rows[0]))
            summary = {"samples": len(rows), "decisions": dict(Counter(
                row["decision"] for row in rows)), "policy": asdict(policy),
                "quality_thresholds": "validation_tail_calibrated_not_quality_validated"
                                      if quality_reference is not None else
                                      "user_supplied_not_verified" if focus_min is not None or
                                      focus_review_min is not None else "not_set",
                "sources": {name: {"path": store_run_path(path, output_dir),
                                    "sha256": file_sha256(path)} for name, path in
                            (("checkpoint", checkpoint), ("reference", reference),
                             ("tiles_npy", tiles_npy))},
                "limitations": metadata["limitations"]}
            if quality_reference is not None:
                summary["quality_reference"] = {
                    "path": store_run_path(quality_reference, output_dir),
                    "sha256": file_sha256(quality_reference)}
                summary["limitations"] += " 本次启用验证尾部质量阈值，非真实不可用检出验证。"
            (stage / "summary.json").write_text(json.dumps(summary, ensure_ascii=False,
                                                          indent=2, allow_nan=False), encoding="utf-8")
        return summary
    finally:
        del tiles
