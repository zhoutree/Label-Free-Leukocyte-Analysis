"""核对清单、图块包和训练统计量是否来自同一份输入。"""

import csv
import hashlib
import json
import warnings
from collections.abc import Iterable, Mapping
from functools import lru_cache
from pathlib import Path

import numpy as np

from .bbbc045_config import TILE_SIZE


@lru_cache(maxsize=16)
def _digest(path: Path, size: int, modified_ns: int) -> str:
    """按文件状态缓存摘要，避免训练与验证重复读取整个图块包。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    """分块计算文件摘要；移动文件不改变摘要值。"""
    resolved = path.resolve()
    state = resolved.stat()
    return _digest(resolved, state.st_size, state.st_mtime_ns)


def pack_metadata_path(tiles_npy: Path) -> Path:
    """返回新图块包旁记录行顺序的元数据路径。"""
    return tiles_npy.with_suffix(tiles_npy.suffix + ".meta.json")


def record_order_sha256(records: Iterable[Mapping[str, str]]) -> str:
    """只对图块身份和顺序取摘要，允许交叉验证改变 split 列。"""
    digest = hashlib.sha256()
    for record in records:
        identity = [record[key] for key in ("file", "donor", "label", "row", "col")]
        digest.update(json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_pack_provenance(split_csv: Path, tiles_npy: Path, *,
                             required: bool = False) -> bool:
    """核对打包顺序与内容；旧包缺元数据时明确提示校验降级。"""
    metadata_path = pack_metadata_path(tiles_npy)
    if not metadata_path.exists():
        if required:
            raise ValueError(f"统计文件要求图块包顺序元数据，但文件缺失：{metadata_path}")
        warnings.warn(
            f"图块包缺少顺序元数据，按旧包兼容模式使用，无法核对原打包顺序：{metadata_path}",
            RuntimeWarning, stacklevel=2,
        )
        return False
    if not metadata_path.is_file():
        raise ValueError(f"图块包顺序元数据不是文件：{metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if set(metadata) != {"record_order_sha256", "packed_tiles_sha256"}:
        raise ValueError("图块包顺序元数据不完整")
    with split_csv.open("r", newline="", encoding="utf-8") as handle:
        order_hash = record_order_sha256(csv.DictReader(handle))
    if order_hash != metadata["record_order_sha256"]:
        raise ValueError("划分清单的图块顺序与打包记录不匹配")
    if file_sha256(tiles_npy) != metadata["packed_tiles_sha256"]:
        raise ValueError("图块包内容与打包记录不匹配")
    return True


def validate_bundle(split_csv: Path, tiles_npy: Path, stats: dict,
                    record_count: int) -> None:
    """先查图块规格，再核对新统计文件中的两份输入摘要。"""
    tiles = np.load(tiles_npy, mmap_mode="r", allow_pickle=False)
    expected = (record_count, TILE_SIZE, TILE_SIZE)
    try:
        if tiles.shape != expected or tiles.dtype != np.float32:
            raise ValueError(f"图块包规格与清单不一致：应为 {expected} float32，"
                             f"实际为 {tiles.shape} {tiles.dtype}")
    finally:
        del tiles

    validate_pack_provenance(
        split_csv, tiles_npy, required=stats.get("pack_metadata_required", False)
    )

    manifest_hash = stats.get("source_manifest_sha256")
    tiles_hash = stats.get("packed_tiles_sha256")
    if (manifest_hash is None) != (tiles_hash is None):
        raise ValueError("训练统计文件的数据指纹不完整")
    if manifest_hash is not None and file_sha256(split_csv) != manifest_hash:
        raise ValueError("训练统计文件与当前划分清单不匹配")
    if tiles_hash is not None and file_sha256(tiles_npy) != tiles_hash:
        raise ValueError("训练统计文件与当前图块包不匹配")
