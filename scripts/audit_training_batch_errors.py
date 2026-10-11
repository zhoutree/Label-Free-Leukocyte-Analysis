"""用已保存的四分类权重回查训练集错分；不修改数据或检查点。"""

from collections import Counter
from pathlib import Path

import torch

from leukocyte_analysis.data.bbbc045_config import CLASS_NAMES
from leukocyte_analysis.data.dataset import LeukocyteDataset
from leukocyte_analysis.evaluation.runner import check_checkpoint_head, load_classifier, predict
from leukocyte_analysis.models.resnet import build_resnet18


PROJECT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT / "data" / "processed"
CHECKPOINT = (
    PROJECT / "runs" / "resnet18_four_imagenet_flips_20261011_143841"
    / "checkpoints" / "best.ckpt"
)


def main() -> None:
    """按受试者、原始类别和批次汇总训练集错分率。"""
    dataset = LeukocyteDataset(
        PROCESSED / "bbbc045_split.csv",
        PROCESSED / "bbbc045_tiles.npy",
        PROCESSED / "bbbc045_train_stats.json",
        "train",
    )
    check_checkpoint_head(CHECKPOINT)
    backbone = build_resnet18(len(CLASS_NAMES), initialization="random")
    model = load_classifier(CHECKPOINT, backbone)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_by_group = Counter()
    wrong_by_group = Counter()
    # 与验证、测试报告共用 Lightning 推理和 CSV 索引关联逻辑。
    for result in predict(model, dataset, batch_size=256, device=device):
        source = dataset.source_by_index[result["packed_index"]]
        group = (
            source["donor"],
            source["source_label"],
            Path(source["file"]).parent.name,
        )
        total_by_group[group] += 1
        if result["predicted_index"] != result["true_index"]:
            wrong_by_group[group] += 1

    print(f"训练集总数 {len(dataset)}；错分 {sum(wrong_by_group.values())}")
    print("至少有 50 个图块的分组，按错分率排序：")
    rows = sorted(
        (
            (wrong_by_group[group] / total, wrong_by_group[group], total, group)
            for group, total in total_by_group.items() if total >= 50
        ),
        reverse=True,
    )
    for rate, errors, total, group in rows[:25]:
        print(f"{group[0]} {group[1]} {group[2]}: {errors}/{total} ({rate:.2%})")


if __name__ == "__main__":
    main()
