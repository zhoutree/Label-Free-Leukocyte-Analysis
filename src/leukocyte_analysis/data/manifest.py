"""扫描 BBBC045 的 ch5 明场 TIFF，建立图块位置与标签的 CSV 清单。

清单只保存文件路径、网格坐标、受试者和类别，不复制或修改图像像素。
"""

import csv
import tempfile
from pathlib import Path

import tifffile

from .bbbc045_config import BRIGHTFIELD_GLOB
from .grid import iter_nonempty_tiles


def find_brightfield_files(data_root: Path) -> list[Path]:
    """查找 BBBC045 数据目录中的所有 ch5 明场 TIFF。"""
    if not data_root.is_dir():
        raise FileNotFoundError(f"数据目录不存在：{data_root}")

    # 目录结构为：受试者 / 细胞类别 / batch / ch5_*.tif。
    files = sorted(data_root.glob(BRIGHTFIELD_GLOB))
    if not files:
        raise FileNotFoundError(f"没有找到 ch5 TIFF：{data_root}")

    return files


def iter_file_rows(tif_path: Path, data_root: Path):
    """逐个读取 TIFF 中的非零格子，生成清单记录。"""
    # 相对路径保存图像位置，前两级目录给出受试者和类别。
    relative_path = tif_path.relative_to(data_root)
    donor, label = relative_path.parts[:2]

    # 保留原始像素，不做显示用的对比度拉伸或归一化。
    image = tifffile.imread(tif_path)

    for row, col, _tile in iter_nonempty_tiles(image):
        yield {
            "file": relative_path.as_posix(),
            "donor": donor,
            "label": label,
            "row": row,
            "col": col,
        }


def write_manifest(data_root: Path, output_csv: Path) -> int:
    """扫描明场 TIFF，成功后才发布完整的清单 CSV。"""
    fieldnames = ["file", "donor", "label", "row", "col"]
    count = 0

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists():
        raise FileExistsError(f"输出文件已存在：{output_csv}")

    # 临时文件与目标文件放在同一磁盘；出错时临时目录自动清理。
    with tempfile.TemporaryDirectory(
        prefix=".manifest-",
        dir=output_csv.parent,
    ) as temporary:
        temporary_csv = Path(temporary) / output_csv.name

        with temporary_csv.open(
            "w", newline="", encoding="utf-8"
        ) as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            # 逐张读取、逐行写入，不把全部细胞图放进内存。
            for tif_path in find_brightfield_files(data_root):
                for record in iter_file_rows(tif_path, data_root):
                    writer.writerow(record)
                    count += 1

        # 文件完整关闭后才发布；Windows 下目标已存在会拒绝覆盖。
        temporary_csv.rename(output_csv)

    return count
