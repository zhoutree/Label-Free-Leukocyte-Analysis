"""与业务无关的报告暂存发布和 CSV 写入，供数据、训练与评估共用。"""

import csv
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO


@contextmanager
def new_report(output_dir: Path, *, prefix: str = ".report-"):
    """在同一磁盘暂存完整报告；失败自动清理，不覆盖已有报告目录。"""
    if output_dir.exists():
        raise FileExistsError(f"报告目录已存在：{output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=prefix, dir=output_dir.parent) as temporary:
        stage = Path(temporary) / "report"
        stage.mkdir()
        yield stage
        stage.rename(output_dir)


def _write_rows(handle: TextIO, rows: list[dict], fields: tuple[str, ...]) -> None:
    """统一 CSV 列顺序和表头；空记录也生成表头。"""
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...], *,
              overwrite: bool = True) -> None:
    """更新表采用原子替换；仅创建模式使用 x，明确拒绝覆盖。

    保留累计频次表的更新语义；只创建的预测表和逐轮记录可传入
    ``overwrite=False``。整个报告目录的事务由 ``new_report`` 管理。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        with path.open("x", newline="", encoding="utf-8") as handle:
            _write_rows(handle, rows, fields)
        return
    staged = None
    try:
        # Windows 下关闭临时文件后再替换，避免打开的句柄阻止重命名。
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", delete=False,
            prefix=".report-csv-", suffix=".csv", dir=path.parent,
        ) as handle:
            staged = Path(handle.name)
            _write_rows(handle, rows, fields)
        staged.replace(path)
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)
