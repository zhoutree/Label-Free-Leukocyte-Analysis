"""从已保存的预测或最佳权重生成可追溯的错分清单。"""

import argparse
import csv
from pathlib import Path

import torch

from .data.bbbc045_config import CLASS_NAMES
from .data.dataset import LeukocyteDataset
from .evaluation.error_records import ERROR_FIELDS, misclassified_rows, write_csv
from .evaluation.runner import check_checkpoint_head, load_classifier, predict
from .models.resnet import build_resnet18


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """允许复用现有预测文件；验证集也可从选定检查点推理一次。"""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions-csv", type=Path, help="已有的逐细胞预测 CSV；无需再次推理")
    source.add_argument("--checkpoint", type=Path, help="仅用于验证集；从该权重生成当前错分快照")
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path,
                        default=PROJECT_ROOT / "data/processed/bbbc045_split.csv")
    parser.add_argument("--tiles-npy", type=Path,
                        default=PROJECT_ROOT / "data/processed/bbbc045_tiles.npy")
    parser.add_argument("--stats-json", type=Path,
                        default=PROJECT_ROOT / "data/processed/bbbc045_train_stats.json")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size 必须大于 0")
    if args.checkpoint and args.split != "val":
        parser.error("测试集只接受既有预测 CSV，避免为审计重复运行测试推理")
    return args


def main() -> None:
    """只保存错分记录，不修改原数据、标签或检查点。"""
    args = parse_args()
    if args.output_csv.exists():
        raise FileExistsError(f"错分记录已存在：{args.output_csv}")
    dataset = LeukocyteDataset(args.split_csv, args.tiles_npy, args.stats_json, args.split)
    if args.predictions_csv:
        with args.predictions_csv.open("r", newline="", encoding="utf-8") as handle:
            predictions = list(csv.DictReader(handle))
    else:
        check_checkpoint_head(args.checkpoint)
        backbone = build_resnet18(len(CLASS_NAMES), initialization="random")
        model = load_classifier(args.checkpoint, backbone)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        predictions = predict(model, dataset, args.batch_size, device)

    expected_indices = [index for index, _label, _donor in dataset.samples]
    actual_indices = [int(row["packed_index"]) for row in predictions]
    if actual_indices != expected_indices:
        raise ValueError("预测文件与当前数据组的 CSV 行号或顺序不一致")

    rows = misclassified_rows(dataset, predictions, args.split)
    write_csv(args.output_csv, rows, ERROR_FIELDS)
    print(f"{args.split} 错分：{len(rows)}；记录：{args.output_csv}")


if __name__ == "__main__":
    main()
