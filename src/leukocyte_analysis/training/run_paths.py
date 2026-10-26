"""在运行记录中保存可迁移路径，并读取旧版绝对路径记录。"""

from pathlib import Path


def store_run_path(path: Path, run_dir: Path) -> str:
    """优先保存相对于运行目录的路径；跨磁盘时保留绝对路径。"""
    absolute = path.resolve()
    base = run_dir.resolve()
    for parent in (base, *base.parents):
        try:
            suffix = absolute.relative_to(parent)
        except ValueError:
            continue
        upward = ("..",) * len(base.relative_to(parent).parts)
        return (Path(*upward) / suffix).as_posix()
    # 不同 Windows 盘符没有共同父目录，保留可供显式迁移的绝对路径。
    return str(absolute)


def resolve_run_path(run_dir: Path, stored: str, override: Path | None = None) -> Path:
    """解析新记录的相对路径，也接受旧记录中的绝对路径。"""
    if override is not None:
        return override.resolve()
    path = Path(stored)
    return path if path.is_absolute() else (run_dir / path).resolve()
