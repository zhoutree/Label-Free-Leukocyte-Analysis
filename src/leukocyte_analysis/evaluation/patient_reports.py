"""核对已有复核结果和患者映射，编排无推理的患者 PDF 导出。"""

import csv
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from ..data.bbbc045_config import CLASS_NAMES, TILE_SIZE
from ..data.integrity import file_sha256
from ..paths import store_run_path
from ..reporting import new_report, write_csv
from .cell_review import ReviewPolicy
from .patient_summary import (CellResult, checked_identifier, group_patients,
                              report_groups, summarize_sample)


def _read_rows(path: Path, required: set[str]) -> list[dict]:
    """读取有明确字段的 CSV；拒绝重复表头与无名多余列。"""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(set(fields)) != len(fields) or not required.issubset(fields):
            raise ValueError(f"CSV 字段缺失或重复：{path.name}")
        rows = list(reader)
    if not rows or any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f"CSV 为空或行字段不完整：{path.name}")
    return rows


def _indexed(rows: list[dict]) -> dict[int, dict]:
    """把明确索引作为主键，不将 CSV 行顺序当作身份。"""
    indexed = {}
    for row in rows:
        index = int(row["packed_index"])
        if index < 0 or index in indexed:
            raise ValueError("packed_index 为负数或重复")
        indexed[index] = row
    return indexed


def load_patient_cells(predictions_csv: Path, patient_manifest: Path,
                       tile_count: int, policy: ReviewPolicy) -> list[CellResult]:
    """要求预测与患者清单完整覆盖同一图块包，并复用原规则检查结果一致性。"""
    required = {"packed_index", "predicted_index", "predicted_label", "decision",
                "confidence", "ood_score", "normalized_tenengrad", "contrast_std",
                "quality_status", "review_reason"}
    predictions = _indexed(_read_rows(predictions_csv, required))
    mapping = _indexed(_read_rows(patient_manifest, {"packed_index", "patient_id", "sample_id"}))
    if set(predictions) != set(range(tile_count)) or set(mapping) != set(predictions):
        raise ValueError("预测和患者清单须完整且不重复地覆盖同一图块包")
    cells = []
    for index, row in predictions.items():
        label = int(row["predicted_index"])
        if label not in range(len(CLASS_NAMES)) or row["predicted_label"] != CLASS_NAMES[label]:
            raise ValueError("预测类别编号与名称不一致")
        focus, score = float(row["normalized_tenengrad"]), float(row["ood_score"])
        assessment = policy.assess(label, score, {"normalized_tenengrad": focus,
                                                  "contrast_std": float(row["contrast_std"])})
        if any(row[key] != assessment[key] for key in
               ("decision", "quality_status", "review_reason")):
            raise ValueError("预测状态与原报告策略不一致，不能生成患者报告")
        owner = mapping[index]
        cells.append(CellResult(
            index, checked_identifier(owner["patient_id"], "patient_id"),
            checked_identifier(owner["sample_id"], "sample_id"),
            checked_identifier(owner.get("cell_id", f"tile-{index}"), "cell_id"),
            label, row["decision"], float(row["confidence"]), score, focus,
            row["quality_status"], row["review_reason"],
        ))
    return cells


def export_patient_reports(review_dir: Path, tiles_npy: Path, patient_manifest: Path,
                           output_dir: Path, *, font_ttf: Path | None = None) -> dict:
    """只读取已有 predict 产物；来源不符或生成失败不发布任何患者 PDF。"""
    if output_dir.exists():
        raise FileExistsError(f"报告目录已存在：{output_dir}")
    source_summary = review_dir / "summary.json"
    predictions = review_dir / "predictions_with_review.csv"
    metadata = json.loads(source_summary.read_text(encoding="utf-8"))
    expected_hash = metadata["sources"]["tiles_npy"]["sha256"]
    if file_sha256(tiles_npy) != expected_hash:
        raise ValueError("图块包与原复核报告的来源摘要不一致")
    policy = ReviewPolicy(**metadata["policy"])
    tiles = np.load(tiles_npy, mmap_mode="r", allow_pickle=False)
    try:
        if (not isinstance(tiles, np.ndarray) or tiles.ndim != 3
                or tiles.shape[1:] != (TILE_SIZE, TILE_SIZE)
                or tiles.dtype != np.float32 or not len(tiles)):
            raise ValueError("患者图块须为非空 N×55×55 float32 原始灰度包")
        cells = load_patient_cells(predictions, patient_manifest, len(tiles), policy)
        patients = group_patients(cells)
        # PDF 依赖延迟导入，原训练和结果统计不依赖 ReportLab。
        try:
            from .patient_pdf import render_patient_pdf, resolve_font
        except ImportError as error:
            raise ImportError('PDF 导出需要先安装 python -m pip install -e ".[pdf]"') from error

        font = resolve_font(font_ttf)
        generated = datetime.now().astimezone().isoformat(timespec="seconds")
        sources = {name: {"path": store_run_path(path, output_dir), "sha256": file_sha256(path)}
                   for name, path in (("review_summary", source_summary),
                                      ("predictions_csv", predictions), ("tiles_npy", tiles_npy),
                                      ("patient_manifest", patient_manifest))}
        result = {"format_version": 1, "generated_at": generated,
                  "pdf_layout": "reference_table_grid_v3",
                  "pdf_explanatory_notes": False,
                  "pdf_percentage_denominator": "all_input_tiles",
                  "conditional_class_percentage_denominator": "accepted_tiles",
                  "cell_details_file": "cell_details.csv",
                  "sources": sources, "review_policy": metadata["policy"],
                  "model_sha256": metadata["sources"]["checkpoint"]["sha256"],
                  "font_file": font.name, "patients": [],
                  "limitations": "研究用途；非临床诊断；PDF 总表与类别栏以全部输入图块为分母；"
                                  "JSON classes.proportion 保留 accepted 内部比例。"
                                  "不是血液绝对浓度，不同样本独立统计。"}
        with new_report(output_dir) as stage:
            details = []
            for ordinal, (patient_id, samples) in enumerate(sorted(patients.items()), 1):
                digest = hashlib.sha256(patient_id.encode("utf-8")).hexdigest()[:12]
                filename = f"patient_{ordinal:04d}_{digest}.pdf"
                summaries = [summarize_sample(sample) for sample in samples.values()]
                render_patient_pdf(stage / filename, patient_id, samples, tiles,
                                   metadata, generated, font)
                result["patients"].append({"patient_id": patient_id, "file": filename,
                                           "samples": summaries})
                for sample in samples.values():
                    for row, selected in report_groups(sample):
                        for position, cell in enumerate(selected, 1):
                            details.append({"pdf_file": filename, "section": row["title"],
                                            "section_position": position, **asdict(cell)})
            write_csv(stage / "cell_details.csv", details, tuple(details[0]), overwrite=False)
            (stage / "summary.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        return result
    finally:
        # 异常回溯或 mock 调用可能仍引用数组；Windows 下显式释放映射。
        mapping = getattr(tiles, "_mmap", None)
        if mapping is not None:
            mapping.close()
        del tiles
