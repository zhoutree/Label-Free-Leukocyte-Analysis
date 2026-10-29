"""将已有图块复核结果导出为每位患者一份 PDF，不加载分类模型。"""

import argparse
from pathlib import Path

from .evaluation.patient_reports import export_patient_reports


def parse_args() -> argparse.Namespace:
    """仅解析结果、原图和患者映射路径；来源核对由业务模块完成。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, required=True,
                        help="review_cells predict 输出，含预测 CSV 与 summary.json")
    parser.add_argument("--tiles-npy", type=Path, required=True, help="该次复核的原始灰度图块包")
    parser.add_argument("--patient-manifest", type=Path, required=True,
                        help="含 packed_index、patient_id、sample_id 的完整 CSV")
    parser.add_argument("--output-dir", type=Path, required=True, help="必须是尚不存在的新目录")
    parser.add_argument("--font-ttf", type=Path, help="可嵌入的中文 TrueType 字体；缺省查找系统字体")
    return parser.parse_args()


def main() -> None:
    """委托患者报告模块；不重新推理、不重新校准或覆盖旧报告。"""
    args = parse_args()
    result = export_patient_reports(args.review_dir, args.tiles_npy, args.patient_manifest,
                                    args.output_dir, font_ttf=args.font_ttf)
    print(f"已生成 {len(result['patients'])} 位患者的 PDF；研究用途，非临床诊断。")
    print("报告目录：", args.output_dir.resolve())


if __name__ == "__main__":
    main()
