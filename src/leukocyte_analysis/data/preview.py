"""从清单中按类别抽样，生成供人眼检查的细胞预览拼图。

亮度拉伸只用于显示，不作为模型训练的预处理。
"""

import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw

from .bbbc045_config import TILE_SIZE


def sample_records(
    manifest_csv: Path, per_class: int = 5, seed: int = 42
) -> dict[str, list[dict[str, str]]]:
    """从清单中为每个类别随机抽取指定数量的格子。"""
    records_by_class = defaultdict(list)

    with manifest_csv.open("r", newline="", encoding="utf-8") as csv_file:
        for record in csv.DictReader(csv_file):
            records_by_class[record["label"]].append(record)

    # 固定种子，使重复预览能抽到相同图块。
    rng = random.Random(seed)
    return {
        label: rng.sample(records, min(per_class, len(records)))
        for label, records in sorted(records_by_class.items())
    }


def _display_range(image: np.ndarray) -> tuple[float, float]:
    """计算一张 TIFF 用于预览的 1%～99% 显示范围。"""
    valid = image[np.isfinite(image) & (image != 0)]
    if valid.size == 0:
        raise ValueError("TIFF 没有可用于预览的有效像素")

    low, high = np.percentile(valid, [1, 99])
    return float(low), float(high)


def _render_tile(
    tile: np.ndarray, low: float, high: float, display_size: int
) -> Image.Image:
    """将原始图块转换为预览图，不修改原始像素。"""
    if high > low:
        display = np.clip((tile - low) / (high - low), 0, 1) * 255
    else:
        display = np.zeros_like(tile)

    tile_image = Image.fromarray(display.astype(np.uint8))
    return tile_image.resize(
        (display_size, display_size),
        Image.Resampling.NEAREST,
    ).convert("RGB")


def make_preview(
    data_root: Path,
    manifest_csv: Path,
    output_png: Path,
    per_class: int = 5,
    tile_size: int = TILE_SIZE,
) -> None:
    """每类抽取若干格子，生成预览拼图。"""
    samples = sample_records(manifest_csv, per_class=per_class)
    labels = list(samples)

    scale = 3
    display_size = tile_size * scale
    label_width = 110
    gap = 10
    row_height = display_size + 35
    width = label_width + per_class * (display_size + gap) + gap
    height = len(labels) * row_height + gap

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    # 先确定每个样本在拼图中的位置，再按源 TIFF 分组读取。
    requests_by_file = defaultdict(list)
    for class_index, label in enumerate(labels):
        y = gap + class_index * row_height
        draw.text((gap, y + display_size // 2), label, fill="black")

        for sample_index, record in enumerate(samples[label]):
            requests_by_file[record["file"]].append(
                (class_index, sample_index, record)
            )

    for relative_path, requests in requests_by_file.items():
        # 同一 TIFF 只解码、计算显示范围一次。
        image = tifffile.imread(data_root / relative_path)
        low, high = _display_range(image)

        for class_index, sample_index, record in requests:
            row, col = int(record["row"]), int(record["col"])
            tile = image[
                row * tile_size : (row + 1) * tile_size,
                col * tile_size : (col + 1) * tile_size,
            ]
            if tile.shape != (tile_size, tile_size):
                raise ValueError(
                    f"图块尺寸异常：{relative_path} ({row}, {col})"
                )

            tile_image = _render_tile(tile, low, high, display_size)
            x = label_width + sample_index * (display_size + gap)
            y = gap + class_index * row_height
            canvas.paste(tile_image, (x, y))
            draw.text(
                (x, y + display_size + 3),
                f'{record["donor"]} ({row},{col})',
                fill="black",
            )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_png)