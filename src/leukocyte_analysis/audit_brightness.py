"""统计 BBBC045 各受试者、原始类别与采集批次的真实亮度分布。"""

import argparse
from pathlib import Path

from .data.intensity_audit import audit_intensity


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-csv", type=Path,
                        default=PROJECT_ROOT / "data/processed/bbbc045_split.csv")
    parser.add_argument("--tiles-npy", type=Path,
                        default=PROJECT_ROOT / "data/processed/bbbc045_tiles.npy")
    parser.add_argument("--include-test", action="store_true",
                        help="事后描述性地纳入 CRF049/CRF066；不得用于选择模型或预处理")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()
    output_dir = args.output_dir or PROJECT_ROOT / "reports" / (
        "bbbc045_brightness_distribution_all_splits" if args.include_test
        else "bbbc045_brightness_distribution"
    )
    report = audit_intensity(args.split_csv, args.tiles_npy, output_dir,
                             chunk_size=args.chunk_size, include_test=args.include_test)
    print(f"已统计 {report['tiles_scanned']} 张图块；报告：{output_dir.resolve()}")


if __name__ == "__main__":
    main()
