"""交叉验证的受试者分折与清单生成；不参与模型训练。"""

import csv
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .bbbc045_config import TEST_DONORS, model_label
from .split import read_manifest as read_manifest


@dataclass(frozen=True)
class DonorFold:
    number: int
    train_donors: frozenset[str]
    val_donors: frozenset[str]


def build_donor_folds(
    records: list[dict[str, str]], n_splits: int, seed: int,
    excluded_donors: frozenset[str] = TEST_DONORS,
) -> list[DonorFold]:
    """用 scikit-learn 分折，确保一个受试者只进入一折验证集。"""
    if n_splits < 2:
        raise ValueError("交叉验证至少需要 2 折")
    eligible = [row for row in records if row["donor"] not in excluded_donors]
    donors = {row["donor"] for row in eligible}
    if n_splits > len(donors):
        raise ValueError("折数不能超过可用受试者人数")

    labels = [model_label(row["label"]) for row in eligible]
    groups = [row["donor"] for row in eligible]
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    seen_validation = set()
    for number, (_train_indices, val_indices) in enumerate(
        splitter.split(np.zeros(len(eligible)), labels, groups), start=1
    ):
        val_donors = frozenset(groups[index] for index in val_indices)
        train_donors = frozenset(donors - val_donors)
        if not val_donors or not train_donors or seen_validation & val_donors:
            raise ValueError("受试者分折不完整或重复")
        seen_validation.update(val_donors)
        folds.append(DonorFold(number, train_donors, val_donors))
    if seen_validation != donors:
        raise ValueError("有受试者没有进入任何验证折")
    return folds


def write_fold_manifest(
    fields: list[str], records: list[dict[str, str]], fold: DonorFold,
    output_csv: Path,
) -> Counter:
    """保留全部原行和索引；测试受试者标为 excluded，不供数据集读取。"""
    if output_csv.exists():
        raise FileExistsError(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with tempfile.TemporaryDirectory(prefix=".fold-", dir=output_csv.parent) as temporary:
        staged = Path(temporary) / output_csv.name
        with staged.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for record in records:
                donor = record["donor"]
                if donor in fold.train_donors:
                    split = "train"
                elif donor in fold.val_donors:
                    split = "val"
                else:
                    split = "excluded"
                writer.writerow({**record, "split": split})
                counts[split] += 1
        staged.rename(output_csv)
    return counts
