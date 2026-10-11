"""对比 CRF049 淋巴与单核明场图块，并检查完全重复；不修改原数据。"""

import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw

from leukocyte_analysis.data.bbbc045_config import TILE_SIZE


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT.parent / "Stained_Montages"
MANIFEST = PROJECT / "data" / "processed" / "bbbc045_manifest.csv"
OUTPUT = PROJECT / "reports" / "previews" / "crf049_lymphocyte_monocyte_comparison.png"
DONOR = "CRF049"
LABELS = ("B", "T", "monocyte")
BATCHES = ("batch1", "batch2")
SAMPLES_PER_ROW = 12
DISPLAY_SIZE = 120
CELL_WIDTH = 132
ROW_HEIGHT = 157
LEFT_MARGIN = 170
TOP_MARGIN = 24


def read_records():
    """按批次与原始类别收集 CRF049 清单行。"""
    groups = defaultdict(list)
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            if record["donor"] != DONOR or record["label"] not in LABELS:
                continue
            batch = Path(record["file"]).parent.name
            if batch in BATCHES:
                groups[(batch, record["label"])].append(record)
    return groups


def main():
    """比较像素完全相同的图块，并保存分组预览。"""
    groups = read_records()
    images = {}

    def get_tile(record):
        relative = record["file"]
        if relative not in images:
            images[relative] = tifffile.imread(SOURCE / relative)
        row, col = int(record["row"]), int(record["col"])
        tile = images[relative][
            row * TILE_SIZE : (row + 1) * TILE_SIZE,
            col * TILE_SIZE : (col + 1) * TILE_SIZE,
        ]
        if tile.shape != (TILE_SIZE, TILE_SIZE):
            raise ValueError(f"Invalid tile: {relative} ({row}, {col})")
        return tile

    def digest(tile):
        return hashlib.blake2b(np.ascontiguousarray(tile).tobytes(), digest_size=16).digest()

    lymphocyte_hashes = defaultdict(list)
    for batch in BATCHES:
        for label in ("B", "T"):
            for record in groups[(batch, label)]:
                lymphocyte_hashes[digest(get_tile(record))].append(record)

    duplicates = []
    for batch in BATCHES:
        for record in groups[(batch, "monocyte")]:
            for other in lymphocyte_hashes.get(digest(get_tile(record)), []):
                if np.array_equal(get_tile(record), get_tile(other)):
                    duplicates.append((record, other))

    selected = {}
    for batch in BATCHES:
        for label in LABELS:
            records = groups[(batch, label)]
            if not records:
                raise ValueError(f"Missing group: {batch}, {label}")
            indices = np.linspace(0, len(records) - 1, min(SAMPLES_PER_ROW, len(records)), dtype=int)
            selected[(batch, label)] = [records[int(index)] for index in indices]

    sample_pixels = np.concatenate(
        [get_tile(record).ravel() for records in selected.values() for record in records]
    )
    low, high = np.percentile(sample_pixels[np.isfinite(sample_pixels) & (sample_pixels != 0)], [1, 99])

    width = LEFT_MARGIN + SAMPLES_PER_ROW * CELL_WIDTH + 12
    height = TOP_MARGIN + len(BATCHES) * len(LABELS) * ROW_HEIGHT + 22
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for batch_index, batch in enumerate(BATCHES):
        for label_index, label in enumerate(LABELS):
            line = batch_index * len(LABELS) + label_index
            y = TOP_MARGIN + line * ROW_HEIGHT
            draw.text((10, y + 35), f"{batch} / {label}", fill="black")
            draw.text((10, y + 53), f"n={len(groups[(batch, label)])}", fill="black")
            for column, record in enumerate(selected[(batch, label)]):
                tile = get_tile(record)
                display = np.clip((tile - low) / (high - low), 0, 1)
                picture = Image.fromarray((display * 255).astype(np.uint8))
                picture = picture.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.Resampling.NEAREST)
                x = LEFT_MARGIN + column * CELL_WIDTH
                canvas.paste(picture.convert("RGB"), (x, y))
                draw.text((x, y + DISPLAY_SIZE + 3), f"r{record['row']} c{record['col']}", fill="black")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT)
    print(f"Saved comparison: {OUTPUT}")
    for batch in BATCHES:
        print(batch, {label: len(groups[(batch, label)]) for label in LABELS})
    print(f"Exact monocyte vs B/T duplicates: {len(duplicates)}")
    for left, right in duplicates[:10]:
        print("duplicate", left, right)


if __name__ == "__main__":
    main()
