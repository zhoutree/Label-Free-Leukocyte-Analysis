"""按划分清单顺序，将明场图块无损打包成可内存映射的数组。"""

import csv
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile

from .bbbc045_config import TILE_SIZE
from .integrity import file_sha256, pack_metadata_path, record_order_sha256


def pack_tiles(
    data_root: Path, split_csv: Path, output_npy: Path
) -> tuple[int, int]:
    """每张 TIFF 只读取一次，并按 CSV 行顺序存储原始图块。

    输出是 ``(图块数, 原始 TIFF 数)``。打包值保持 float32，不做归一化；
    全部完成后才发布输出文件，已有文件不会被覆盖。
    """
    with split_csv.open("r", newline="", encoding="utf-8") as csv_file:
        records = list(csv.DictReader(csv_file))

    if not records:
        raise ValueError("划分清单为空")

    # CSV 行号就是打包数组中的索引；按原始 TIFF 分组以减少读取。
    by_file = defaultdict(list)
    for index, record in enumerate(records):
        by_file[record["file"]].append(
            (index, int(record["row"]), int(record["col"]))
        )

    output_npy.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = pack_metadata_path(output_npy)
    for target in (output_npy, metadata_path):
        if target.exists():
            raise FileExistsError(f"输出文件已存在：{target}")

    # 临时文件与正式文件位于同一磁盘，失败时自动清理。
    with tempfile.TemporaryDirectory(
        prefix=".pack-",
        dir=output_npy.parent,
    ) as temporary:
        temporary_npy = Path(temporary) / output_npy.name
        tiles = np.lib.format.open_memmap(
            temporary_npy,
            mode="w+",
            dtype=np.float32,
            shape=(len(records), TILE_SIZE, TILE_SIZE),
        )

        try:
            for file_number, (relative_path, positions) in enumerate(
                by_file.items(), start=1
            ):
                # 同一张 montage 只解码一次。
                image = tifffile.imread(data_root / relative_path)
                if image.dtype != np.float32:
                    raise ValueError(f"不是 float32：{relative_path}")

                for index, row, col in positions:
                    tile = image[
                        row * TILE_SIZE : (row + 1) * TILE_SIZE,
                        col * TILE_SIZE : (col + 1) * TILE_SIZE,
                    ]
                    if tile.shape != (TILE_SIZE, TILE_SIZE):
                        raise ValueError(
                            f"格子尺寸异常：{relative_path} ({row}, {col})"
                        )
                    if not np.isfinite(tile).all() or not np.any(tile != 0):
                        raise ValueError(
                            f"格子像素异常：{relative_path} ({row}, {col})"
                        )

                    # 保留原始 float32 数值，不做显示拉伸或归一化。
                    tiles[index] = tile

                if file_number % 25 == 0:
                    print(f"已处理 {file_number}/{len(by_file)} 张 TIFF")

            tiles.flush()
        finally:
            # Windows 清理临时目录前，先释放内存映射文件。
            del tiles

        # 顺序与内容绑定到新包，供后续统计、训练和交叉验证核对。
        metadata = {
            "record_order_sha256": record_order_sha256(records),
            "packed_tiles_sha256": file_sha256(temporary_npy),
        }
        temporary_metadata = Path(temporary) / metadata_path.name
        temporary_metadata.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # 图块已全部写完并释放映射后才发布。
        temporary_metadata.rename(metadata_path)
        try:
            temporary_npy.rename(output_npy)
        except OSError:
            # 只回滚本次刚发布的元数据；图块仍在临时目录中自动清理。
            metadata_path.unlink()
            raise

    return len(records), len(by_file)


def verify_pack(
    data_root: Path,
    split_csv: Path,
    packed_npy: Path,
    checks: int = 10,
) -> None:
    """按固定随机种子抽查原始 TIFF 和打包图块的逐像素一致性。

    除 ``checks`` 个随机位置外，始终检查清单首尾；不改写任何数据。
    """
    with split_csv.open("r", newline="", encoding="utf-8") as csv_file:
        records = list(csv.DictReader(csv_file))

    if not records:
        raise ValueError("划分清单为空")

    tiles = np.load(packed_npy, mmap_mode="r", allow_pickle=False)
    expected_shape = (len(records), TILE_SIZE, TILE_SIZE)
    if tiles.shape != expected_shape or tiles.dtype != np.float32:
        raise ValueError(f"打包数组规格异常：{tiles.shape}，{tiles.dtype}")

    # 固定随机种子，重复运行时检查同一批位置；另含首尾记录。
    rng = np.random.default_rng(42)
    indices = {0, len(records) - 1}
    indices.update(
        rng.choice(
            len(records),
            size=min(checks, len(records)),
            replace=False,
        ).tolist()
    )

    for index in sorted(indices):
        record = records[index]
        image = tifffile.imread(data_root / record["file"])
        row, col = int(record["row"]), int(record["col"])
        original = image[
            row * TILE_SIZE : (row + 1) * TILE_SIZE,
            col * TILE_SIZE : (col + 1) * TILE_SIZE,
        ]
        if not np.array_equal(tiles[index], original):
            raise ValueError(f"第 {index} 条记录与原图不一致")

    print(f"验证通过：{len(indices)} 个图块与原始 TIFF 完全一致")
