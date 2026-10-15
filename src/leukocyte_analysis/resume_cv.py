"""续训指定交叉验证折，不重跑其他折。"""

import argparse
from datetime import datetime
from pathlib import Path

from .training.cv_resume import resume_cv_fold


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    """解析单折续训参数，并打印续训后选用的检查点结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True,
                        help="续训后的总轮数上限，不是额外增加的轮数")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.fold < 1 or args.epochs < 1:
        parser.error("fold 和 epochs 必须大于 0")
    output_dir = args.output_dir or (
        PROJECT_ROOT / "runs" / f"cv_fold{args.fold:02d}_extended_{datetime.now():%Y%m%d_%H%M%S}"
    )
    summary = resume_cv_fold(args.source_run, output_dir,
                             fold_number=args.fold, epochs=args.epochs)
    print("第", args.fold, "折续训后选用：", summary["continuation"]["selected"])
    print(f"折外准确率：{summary['out_of_fold']['accuracy']:.4f}；"
          f"macro-F1：{summary['out_of_fold']['macro_f1']:.4f}")
    print("结果目录：", output_dir)


if __name__ == "__main__":
    main()
