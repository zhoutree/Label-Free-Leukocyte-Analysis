"""从已保存的预测或最佳权重生成可追溯的错分清单。"""

import argparse
from pathlib import Path

from .evaluation.error_export import export_errors
from .paths import PROJECT_ROOT


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
    count = export_errors(args.split_csv, args.tiles_npy, args.stats_json, args.split,
                          args.output_csv, predictions_csv=args.predictions_csv,
                          checkpoint=args.checkpoint, batch_size=args.batch_size)
    print(f"{args.split} 错分：{count}；记录：{args.output_csv}")


if __name__ == "__main__":
    main()
