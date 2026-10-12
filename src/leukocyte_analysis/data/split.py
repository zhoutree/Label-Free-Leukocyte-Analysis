"""按受试者划分训练、验证和测试集，并生成带 split 列的清单。"""

import csv
import tempfile
from collections import Counter
from pathlib import Path

from .bbbc045_config import TEST_DONORS, TRAIN_DONORS, VAL_DONORS


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """读取带 split 列的图块清单，并检查必需列与记录。"""
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"file", "donor", "label", "row", "col", "split"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("划分清单缺少必需列")
        fields = list(reader.fieldnames)
        records = list(reader)
    if not records:
        raise ValueError("划分清单为空")
    return fields, records


def split_for_donor(donor: str) -> str:
    """根据受试者 ID 返回 train、val 或 test。"""
    if donor in TRAIN_DONORS:
        return "train"
    if donor in VAL_DONORS:
        return "val"
    if donor in TEST_DONORS:
        return "test"
    raise ValueError(f"未知受试者 ID：{donor}")


def write_split_manifest(manifest_csv: Path, output_csv: Path) -> Counter:
    """成功处理全部记录后，才发布带 split 列的清单。"""
    # 同一受试者不能同时出现在两个数据组。
    if (
        TRAIN_DONORS & VAL_DONORS
        or TRAIN_DONORS & TEST_DONORS
        or VAL_DONORS & TEST_DONORS
    ):
        raise ValueError("训练、验证、测试受试者名单有重叠")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists():
        raise FileExistsError(f"输出文件已存在：{output_csv}")

    counts = Counter()

    # 失败时临时目录自动清理，不留下半成品正式文件。
    with tempfile.TemporaryDirectory(
        prefix=".split-",
        dir=output_csv.parent,
    ) as temporary:
        temporary_csv = Path(temporary) / output_csv.name

        with manifest_csv.open(
            "r", newline="", encoding="utf-8"
        ) as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None or "donor" not in reader.fieldnames:
                raise ValueError("原清单缺少 donor 列")
            if "split" in reader.fieldnames:
                raise ValueError("原清单已经包含 split 列")

            with temporary_csv.open(
                "w", newline="", encoding="utf-8"
            ) as target:
                writer = csv.DictWriter(
                    target,
                    fieldnames=[*reader.fieldnames, "split"],
                )
                writer.writeheader()

                for record in reader:
                    split = split_for_donor(record["donor"])
                    record["split"] = split
                    writer.writerow(record)
                    counts[split] += 1

        # 临时 CSV 已完整关闭后才发布。
        temporary_csv.rename(output_csv)

    return counts
