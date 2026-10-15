"""调用现有数据模块，将 BBBC045 原始明场 montage 准备为训练输入。"""

import argparse
from pathlib import Path

from .data.manifest import write_manifest
from .data.pack import pack_tiles, verify_pack
from .data.split import write_split_manifest
from .data.stats import write_train_stats


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data/processed"


def parse_args() -> argparse.Namespace:
    """读取原图根目录与处理结果目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, required=True,
        help="解压后直接包含 CRF* 子目录的路径",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="输出四个训练输入文件；默认 data/processed/",
    )
    return parser.parse_args()


def prepare_data(data_root: Path, output_dir: Path) -> None:
    """生成清单、固定受试者划分、图块包和训练侧统计量。"""
    manifest = output_dir / "bbbc045_manifest.csv"
    split = output_dir / "bbbc045_split.csv"
    tiles = output_dir / "bbbc045_tiles.npy"
    stats = output_dir / "bbbc045_train_stats.json"

    # 在耗时的 TIFF 处理前核对所有目标，避免覆盖或形成新旧文件混合。
    existing = [path for path in (manifest, split, tiles, stats) if path.exists()]
    if existing:
        raise FileExistsError(
            "以下输出已存在，请选择新的 --output-dir，勿覆盖现有数据："
            + ", ".join(str(path) for path in existing)
        )

    count = write_manifest(data_root, manifest)
    counts = write_split_manifest(manifest, split)
    packed_count, file_count = pack_tiles(data_root, split, tiles)
    verify_pack(data_root, split, tiles)
    train_stats = write_train_stats(split, tiles, stats)
    print(
        f"准备完成：{count} 个清单图块，{packed_count} 个打包图块，"
        f"{file_count} 张明场 TIFF；划分 {dict(counts)}；"
        f"训练统计使用 {train_stats['train_tiles']} 个图块。"
    )
    print(f"输出目录：{output_dir.resolve()}")


def main() -> None:
    """执行数据准备命令。"""
    args = parse_args()
    prepare_data(args.data_root, args.output_dir)


if __name__ == "__main__":
    main()
