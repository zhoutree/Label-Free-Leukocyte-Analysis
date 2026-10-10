"""用最佳验证检查点对保留的 BBBC045 测试受试者做一次最终评估。"""

import argparse
from pathlib import Path

from .evaluation.runner import evaluate_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """读取检查点和测试数据路径；不修改训练设置。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="已确定模型的 checkpoints/best.ckpt")
    parser.add_argument("--split-csv", type=Path,
                        default=PROJECT_ROOT / "data/processed/bbbc045_split.csv")
    parser.add_argument("--tiles-npy", type=Path,
                        default=PROJECT_ROOT / "data/processed/bbbc045_tiles.npy")
    parser.add_argument("--stats-json", type=Path,
                        default=PROJECT_ROOT / "data/processed/bbbc045_train_stats.json")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size 必须大于 0")
    return args


def main() -> None:
    """把评估流程与命令行接口分开，便于测试和复用。"""
    args = parse_args()
    output_dir = args.output_dir or args.checkpoint.resolve().parent.parent / "test_evaluation"
    summary = evaluate_checkpoint(args.checkpoint, args.split_csv, args.tiles_npy,
                                  args.stats_json, output_dir, args.batch_size)
    print(f"测试细胞：{summary['samples']}；准确率：{summary['accuracy']:.4f}；"
          f"macro-F1：{summary['macro_f1']:.4f}")
    print("评估报告：", output_dir)


if __name__ == "__main__":
    main()
