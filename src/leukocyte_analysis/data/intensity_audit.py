"""描述明场图块的实际亮度分布，不修改图像或模型。"""

import csv
import json
import tempfile
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .bbbc045_config import TEST_DONORS, TILE_SIZE, TRAIN_DONORS, VAL_DONORS
from .split import read_manifest, split_for_donor


BORDER_WIDTH = 5


def _batch_name(record: dict[str, str]) -> str:
    """从经核对的原 TIFF 相对路径提取采集批次名称。"""
    parts = record["file"].replace("\\", "/").split("/")
    if (len(parts) != 4 or parts[0] != record["donor"]
            or parts[1] != record["label"] or not parts[2].startswith("batch")):
        raise ValueError(f"原始 TIFF 路径与清单不一致：{record['file']}")
    return parts[2]


def _tile_metrics(block: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """整图均值、外侧 5 像素环中位数，以及图内 P90−P10。"""
    if block.ndim != 3 or block.shape[1:] != (TILE_SIZE, TILE_SIZE) or not np.isfinite(block).all():
        raise ValueError("图块形状异常或含非有限像素")
    # 外缘环只是照明代理；细胞触边时也会混入细胞像素，不能视为纯背景。
    border = np.ones((TILE_SIZE, TILE_SIZE), dtype=bool)
    border[BORDER_WIDTH:-BORDER_WIDTH, BORDER_WIDTH:-BORDER_WIDTH] = False
    flat = block.reshape(len(block), -1)
    p10, p90 = np.percentile(flat, (10, 90), axis=1)
    return flat.mean(axis=1), np.median(block[:, border], axis=1), p90 - p10


def _write_csv(path: Path, rows: list[dict]) -> None:
    """将非空审计记录按首行字段顺序写入暂存 CSV。"""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    """按给定分组报告图块强度分布，不对任一组做质量筛除。"""
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    summary = []
    for group_key, group in sorted(groups.items()):
        result = {**dict(zip(keys, group_key)), "tiles": len(group)}
        for key in ("tile_mean", "border_median_proxy", "tile_p90_minus_p10"):
            values = np.array([row[key] for row in group], dtype=np.float64)
            result[f"{key}_median"] = float(np.median(values))
            result[f"{key}_p10"] = float(np.percentile(values, 10))
            result[f"{key}_p90"] = float(np.percentile(values, 90))
        summary.append(result)
    return summary


def _compare_batches(groups: list[dict]) -> list[dict]:
    """对同一受试者、同一原始类别的批次计算 B/A 比值及差值。"""
    by_donor_class = defaultdict(list)
    for row in groups:
        by_donor_class[(row["donor"], row["label"])].append(row)
    comparisons = []
    for (donor, label), batches in sorted(by_donor_class.items()):
        for first, second in combinations(sorted(batches, key=lambda row: row["batch"]), 2):
            values = {"split": first["split"], "donor": donor, "label": label,
                      "batch_a": first["batch"], "batch_b": second["batch"],
                      "tiles_a": first["tiles"], "tiles_b": second["tiles"]}
            for key in ("tile_mean", "border_median_proxy", "tile_p90_minus_p10"):
                a, b = first[f"{key}_median"], second[f"{key}_median"]
                values[f"{key}_ratio_b_over_a"] = b / a if a > 0 else float("nan")
                values[f"{key}_delta_b_minus_a"] = b - a
            comparisons.append(values)
    return comparisons


def _save_plot(comparisons: list[dict], output: Path) -> None:
    """仅展示双方至少 50 张的最大 20 组差异；不影响 CSV 或训练。"""
    eligible = [row for row in comparisons if min(row["tiles_a"], row["tiles_b"]) >= 50]
    eligible.sort(key=lambda row: abs(row["border_median_proxy_ratio_b_over_a"] - 1),
                  reverse=True)
    selected = eligible[:20]
    if not selected:
        return
    labels = [f"{row['split']}: {row['donor']}/{row['label']} "
              f"({row['batch_b']} ÷ {row['batch_a']})"
              for row in selected]
    shifts = [100 * (row["border_median_proxy_ratio_b_over_a"] - 1)
              for row in selected]
    fig, axis = plt.subplots(figsize=(11, max(5, len(selected) * 0.35)),
                             constrained_layout=True)
    axis.barh(labels[::-1], shifts[::-1])
    axis.axvline(0, color="black", linewidth=0.7)
    axis.set(xlabel="Border-median proxy shift from batch A to B (%)",
             title="Largest within-donor/class batch shifts (at least 50 tiles per batch)")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def audit_intensity(
    split_csv: Path, tiles_npy: Path, output_dir: Path, *, chunk_size: int = 256,
    include_test: bool = False,
) -> dict:
    """按清单行号分批扫描明场强度，并原子发布描述性报告。

    默认只统计原训练／验证受试者；``include_test`` 必须显式开启，
    纳入测试集也仅供事后分析，不可据此选择模型或预处理参数。
    返回报告设置，不修改原始图块、标签或划分。
    """
    if chunk_size < 1:
        raise ValueError("chunk-size 必须大于 0")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    _fields, records = read_manifest(split_csv)
    tiles = np.load(tiles_npy, mmap_mode="r", allow_pickle=False)
    if len(records) != len(tiles):
        raise ValueError("清单行数与图块数不一致")
    eligible = TRAIN_DONORS | VAL_DONORS | (TEST_DONORS if include_test else frozenset())
    if any(row["split"] != split_for_donor(row["donor"]) for row in records):
        raise ValueError("受试者的原始 train/val/test 划分与配置不一致")
    indices = [i for i, row in enumerate(records) if row["donor"] in eligible]
    if not indices:
        raise ValueError("没有找到可审计的图块")

    tile_rows = []
    for start in range(0, len(indices), chunk_size):
        chunk = indices[start:start + chunk_size]
        mean, border, contrast = _tile_metrics(np.asarray(tiles[chunk], dtype=np.float32))
        for position, index in enumerate(chunk):
            source = records[index]
            tile_rows.append({
                "packed_index": index, "split": source["split"],
                "donor": source["donor"], "label": source["label"],
                "batch": _batch_name(source), "file": source["file"],
                "tile_mean": float(mean[position]),
                "border_median_proxy": float(border[position]),
                "tile_p90_minus_p10": float(contrast[position]),
            })
        if start % (chunk_size * 20) == 0:
            print(f"已扫描 {min(start + chunk_size, len(indices))}/{len(indices)} 张", flush=True)
    del tiles

    groups = _summarize(tile_rows, ("split", "donor", "label", "batch"))
    split_summary = _summarize(tile_rows, ("split",))
    comparisons = _compare_batches(groups)
    settings = {
        "purpose": "事后描述真实明场强度分布；不得用测试集选择预处理或模型",
        "split_csv": str(split_csv.resolve()), "tiles_npy": str(tiles_npy.resolve()),
        "included_donors": sorted(eligible),
        "excluded_donors": sorted(TEST_DONORS - eligible),
        "include_test": include_test,
        "split_counts": dict(sorted(Counter(row["split"] for row in tile_rows).items())),
        "tiles_scanned": len(tile_rows),
        "border_proxy": f"{TILE_SIZE}×{TILE_SIZE} 图块外侧 {BORDER_WIDTH} 像素环的中位数；细胞触边时并非纯背景",
        "contrast_proxy": "单图块像素 P90−P10；不等于形态学对比度",
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".intensity-", dir=output_dir.parent) as temporary:
        staged = Path(temporary) / "report"
        staged.mkdir()
        _write_csv(staged / "tile_metrics.csv", tile_rows)
        _write_csv(staged / "group_summary.csv", groups)
        _write_csv(staged / "split_summary.csv", split_summary)
        if comparisons:
            _write_csv(staged / "within_donor_batch_pairs.csv", comparisons)
            _save_plot(comparisons, staged / "largest_batch_shifts.png")
        (staged / "settings.json").write_text(
            json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        staged.rename(output_dir)
    return settings
