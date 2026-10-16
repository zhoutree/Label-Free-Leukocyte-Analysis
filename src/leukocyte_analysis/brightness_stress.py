"""比较两组已完成交叉验证权重在亮度扰动下的折外表现。"""

import argparse
from pathlib import Path

from .evaluation.brightness import compare_brightness


def main() -> None:
    """从命令行比较两组交叉验证权重的折外亮度敏感度。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--augmented-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tiles-npy", type=Path, help="迁移旧运行后使用的新图块位置")
    parser.add_argument("--factors", type=float, nargs="+", default=(0.8, 1.0, 1.2))
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    rows = compare_brightness(args.baseline_run, args.augmented_run, args.output_dir,
                              factors=tuple(args.factors), batch_size=args.batch_size,
                              tiles_npy=args.tiles_npy)
    for row in rows:
        print(f"{row['run']} ×{row['brightness_factor']:g}: "
              f"accuracy={row['accuracy']:.4f}, macro-F1={row['macro_f1']:.4f}")
    print("诊断报告：", args.output_dir.resolve())


if __name__ == "__main__":
    main()
