"""用合成结果检查患者分组、分母、来源隔离和 PDF 导出；不加载模型。"""

import csv
import json
from importlib.util import find_spec
from pathlib import Path
import re
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from leukocyte_analysis.data.bbbc045_config import CLASS_NAMES
from leukocyte_analysis.data.integrity import file_sha256
from leukocyte_analysis.evaluation.cell_review import ReviewPolicy
from leukocyte_analysis.evaluation.patient_reports import export_patient_reports, load_patient_cells
from leukocyte_analysis.evaluation.patient_summary import CellResult, group_patients, summarize_sample
from leukocyte_analysis.patient_report import main, parse_args
from leukocyte_analysis.reporting import write_csv


def sample_cell(index: int = 0, *, patient: str = "P001", sample: str = "S001",
                decision: str = "accepted", predicted: int = 0) -> CellResult:
    """建立不含图像或模型的有效细胞记录。"""
    reason = {"accepted": "", "needs_review": "feature_outlier",
              "unusable": "severe_blur_candidate"}[decision]
    return CellResult(index, patient, sample, f"cell-{index}", predicted, decision,
                      .9, 2.0 if decision == "needs_review" else 0.0,
                      2.0 if decision == "unusable" else 10.0,
                      "unusable" if decision == "unusable" else "passed", reason)


class TestPatientSummary(unittest.TestCase):
    """通过、待复核和不可用互斥；同患者不同样本不能合并计算比例。"""

    def test_denominator_excludes_review_and_unusable(self) -> None:
        """只统计 accepted 候选，两个拦截项不改变四类比例。"""
        cells = [sample_cell(0), sample_cell(1, predicted=2),
                 sample_cell(2, decision="needs_review"), sample_cell(3, decision="unusable")]
        result = summarize_sample(cells)
        self.assertEqual(result["decisions"], {"accepted": 2, "needs_review": 1, "unusable": 1})
        self.assertEqual([r["proportion"] for r in result["classes"]], [.5, 0, .5, 0])
        self.assertEqual(sum(r["count"] for r in result["report_rows"]), 4)
        self.assertAlmostEqual(sum(r["proportion"] for r in result["report_rows"]), 1)
        report = {row["key"]: row for row in result["report_rows"]}
        self.assertEqual(report["lymphocyte"]["proportion"], .25)
        self.assertEqual(report["needs_review"]["count"], 1)
        self.assertEqual(report["unusable"]["proportion"], .25)
        empty = summarize_sample([sample_cell(decision="needs_review")])
        self.assertTrue(all(row["proportion"] is None for row in empty["classes"]))
        self.assertEqual(empty["report_rows"][-2]["proportion"], 1)

    def test_grouping_separates_samples_and_refuses_ambiguous_ownership(self) -> None:
        """不同样本分组，拒绝一份标本跨患者及重复图块／样本内编号。"""
        cells = [sample_cell(0), sample_cell(1, sample="S002")]
        self.assertEqual(set(group_patients(cells)["P001"]), {"S001", "S002"})
        for invalid in ([cells[0], sample_cell(1, patient="P002")],
                        [cells[0], cells[0]], []):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                group_patients(invalid)
        with self.assertRaises(ValueError):
            summarize_sample(cells)

    def test_identifiers_and_nonfinite_scores_are_rejected(self) -> None:
        """换行、超长编号和非有限数不能进入报告。"""
        for patient in ("", "P\n001", "P" * 65):
            with self.subTest(patient=patient), self.assertRaises(ValueError):
                sample_cell(patient=patient)
        with self.assertRaises(ValueError):
            CellResult(0, "P", "S", "C", 0, "accepted", float("nan"), 0, 1, "passed", "")


class TestPatientReportExport(unittest.TestCase):
    """用很小的包和现成决策验证 CSV 输入契约及原子报告发布。"""

    def setUp(self) -> None:
        """合成两个患者、三个样本；临时文件全部由本测试拥有。"""
        self.temporary = tempfile.TemporaryDirectory(prefix="patient-report-test-",
                                                     dir=Path(__file__).resolve().parent)
        self.addCleanup(self.cleanup_temporary)
        self.root = Path(self.temporary.name)
        self.review = self.root / "review"
        self.review.mkdir()
        self.tiles = self.root / "tiles.npy"
        np.save(self.tiles, np.tile(np.arange(55 * 55, dtype=np.float32).reshape(1, 55, 55),
                                   (4, 1, 1)))
        self.manifest = self.root / "patients.csv"
        self.mapping = [dict(packed_index=i, patient_id="P001" if i < 3 else "P002",
                             sample_id="S001" if i < 2 else f"S00{i}", cell_id=f"cell-{i}")
                        for i in range(4)]
        self.policy = ReviewPolicy(1, focus_min=5)
        self.rows = []
        for i in range(4):
            quality = {"normalized_tenengrad": 2 if i == 2 else 10, "contrast_std": 1}
            score = 2 if i == 1 else 0
            self.rows.append({"packed_index": i, "predicted_index": i,
                              "predicted_label": CLASS_NAMES[i], "confidence": .9,
                              "ood_score": score, **quality,
                              **self.policy.assess(i, score, quality)})
        self.metadata = {"samples": 4, "policy": {"ood_threshold": 1, "focus_min": 5,
                                                  "focus_review_min": None},
                         "sources": {"tiles_npy": {"sha256": file_sha256(self.tiles)},
                                     "checkpoint": {"sha256": "a" * 64}}}
        self.write_inputs()

    def cleanup_temporary(self) -> None:
        """只对 Windows 短暂占用／延迟删除做有限重试，其他清理失败照常报错。"""
        for attempt in range(4):
            try:
                self.temporary.cleanup()
                return
            except OSError as error:
                if getattr(error, "winerror", None) not in {32, 145} or attempt == 3:
                    raise
                time.sleep(.05 * (attempt + 1))

    def write_inputs(self) -> None:
        """更新当前测试自己创建的三份输入，不涉及正式报告。"""
        write_csv(self.manifest, self.mapping, tuple(self.mapping[0]))
        write_csv(self.review / "predictions_with_review.csv", self.rows, tuple(self.rows[0]))
        (self.review / "summary.json").write_text(json.dumps(self.metadata), encoding="utf-8")

    def test_csv_join_uses_indices_and_rejects_missing_duplicate_or_wrong_rows(self) -> None:
        """清单乱序可读取；缺索引、重复索引和错误状态必须拒绝。"""
        self.mapping.reverse()
        self.write_inputs()
        loaded = load_patient_cells(self.review / "predictions_with_review.csv",
                                    self.manifest, 4, self.policy)
        self.assertEqual(loaded[0].sample_id, "S001")
        original = list(self.mapping)
        for invalid in (original[:-1], [original[0], *original[:-1]]):
            self.mapping = invalid
            self.write_inputs()
            with self.assertRaises(ValueError):
                load_patient_cells(self.review / "predictions_with_review.csv",
                                   self.manifest, 4, self.policy)
        self.mapping = original
        self.rows[1]["decision"] = "accepted"
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "状态"):
            load_patient_cells(self.review / "predictions_with_review.csv", self.manifest, 4,
                               self.policy)

    def test_wrong_pack_refused_without_creating_output(self) -> None:
        """即使图块形状相同，内容摘要不一致也不能与旧预测配对。"""
        np.save(self.tiles, np.zeros((4, 55, 55), dtype=np.float32))
        output = self.root / "out"
        with self.assertRaisesRegex(ValueError, "摘要"):
            export_patient_reports(self.review, self.tiles, self.manifest, output)
        self.assertFalse(output.exists())

    @unittest.skipUnless(find_spec("reportlab"), "未安装可选 PDF 依赖")
    def test_pagination_and_zero_accepted_quality_unassessed(self) -> None:
        """紧凑大图册自动续页；零 accepted 样本的总表仍以全部图块统计。"""
        from leukocyte_analysis.evaluation.patient_pdf import render_patient_pdf, resolve_font

        try:
            font = resolve_font()
        except ValueError:
            self.skipTest("无可嵌入的系统中文字体")
        cells = [CellResult(i, "P001", "S001", f"C{i}", 0, "needs_review", .9,
                            2, 10, "not_assessed", "feature_outlier") for i in range(320)]
        self.metadata["policy"]["focus_min"] = None
        path = self.root / "pagination.pdf"
        render_patient_pdf(path, "P001", {"S001": cells}, np.zeros((320, 55, 55), np.float32),
                           self.metadata, "2026-10-28T12:00:00+08:00", font)
        self.assertGreater(len(re.findall(rb"/Type\s*/Page\b", path.read_bytes())), 2)
        self.assertEqual(summarize_sample(cells)["quality_not_assessed"], 320)

    @unittest.skipUnless(find_spec("reportlab"), "未安装可选 PDF 依赖")
    def test_reference_grid_has_ten_48_point_images_without_captions(self) -> None:
        """严格复用参考 PDF 的十列位置与方图尺寸，逐图明细不占用图册间距。"""
        from leukocyte_analysis.evaluation.patient_pdf import _gallery_row

        canvas = Mock()
        cells = [sample_cell(i) for i in range(10)]
        _gallery_row(canvas, cells, np.zeros((10, 55, 55), np.float32), 700)
        self.assertEqual(canvas.drawImage.call_count, 10)
        positions = [call.args[1:] for call in canvas.drawImage.call_args_list]
        self.assertEqual(positions[0], (31.18, 649.17, 48, 48))
        self.assertAlmostEqual(positions[-1][0], 514.21)
        canvas.drawString.assert_not_called()

    @unittest.skipUnless(find_spec("reportlab"), "未安装可选 PDF 依赖")
    def test_pdf_has_no_explanatory_notes_or_disclaimer(self) -> None:
        """纯结果版不在总表下或页脚追加说明，即使质量未评估也只保留原始状态。"""
        from leukocyte_analysis.evaluation.patient_pdf import _header, _summary

        canvas = Mock()
        cell = CellResult(0, "P001", "S001", "C001", 0, "accepted", .9,
                          0, 10, "not_assessed", "")
        with patch("leukocyte_analysis.evaluation.patient_pdf._paragraph") as paragraph:
            _summary(canvas, [cell], "font", 700)
            paragraph.assert_not_called()
            paragraph.return_value = 700
            _header(canvas, "font", "P001", "S001", "2026-10-29T12:00:00+08:00")
        text = "\n".join(str(call.args[-1]) for method in
                         (canvas.drawString, canvas.drawRightString, canvas.drawCentredString)
                         for call in method.call_args_list)
        for phrase in ("研究用途", "非临床诊断", "待复核不等于", "占比均以", "cell_details.csv",
                       "not_assessed"):
            self.assertNotIn(phrase, text)
        self.assertIn("患者：P001", text)
        self.assertIn("输入图块总数", text)

    @unittest.skipUnless(find_spec("reportlab"), "未安装可选 PDF 依赖")
    def test_explicit_font_and_nonfinite_tile_checks(self) -> None:
        """不存在的字体或含非有限像素的图片不能静默变成缺字／空白报告。"""
        from leukocyte_analysis.evaluation.patient_pdf import _display_image, resolve_font

        with self.assertRaisesRegex(ValueError, "字体"):
            resolve_font(self.root / "missing.ttf")
        with self.assertRaisesRegex(ValueError, "非有限"):
            _display_image(np.full((55, 55), np.nan, np.float32))

    @unittest.skipUnless(find_spec("reportlab"), "未安装可选 PDF 依赖")
    def test_export_per_patient_multi_sample_and_no_overwrite(self) -> None:
        """一位患者一份 PDF、样本分开，长编号与特殊字符不影响安全命名。"""
        from leukocyte_analysis.evaluation.patient_pdf import resolve_font

        try:
            resolve_font()
        except ValueError:
            self.skipTest("无可嵌入的系统中文字体")
        for owner in self.mapping[:3]:
            owner["patient_id"] = "患者/<演示>&" * 6
        self.write_inputs()
        output = self.root / "out"
        result = export_patient_reports(self.review, self.tiles, self.manifest, output)
        self.assertEqual(len(result["patients"]), 2)
        totals = sorted(len(patient["samples"]) for patient in result["patients"])
        self.assertEqual(totals, [1, 2])
        self.assertEqual(result["pdf_percentage_denominator"], "all_input_tiles")
        self.assertFalse(result["pdf_explanatory_notes"])
        with (output / "cell_details.csv").open(newline="", encoding="utf-8") as handle:
            details = list(csv.DictReader(handle))
        self.assertEqual(len(details), 4)
        self.assertEqual({int(row["packed_index"]) for row in details}, set(range(4)))
        self.assertEqual({row["section"] for row in details},
                         {"淋巴细胞", "待复核", "质量不合格", "中性粒细胞"})
        for patient in result["patients"]:
            self.assertRegex(patient["file"], r"^patient_\d{4}_[a-f0-9]{12}\.pdf$")
            data = (output / patient["file"]).read_bytes()
            self.assertTrue(data.startswith(b"%PDF"))
            self.assertGreaterEqual(len(re.findall(rb"/Type\s*/Page\b", data)), len(patient["samples"]))
        before = (output / "summary.json").read_bytes()
        with self.assertRaises(FileExistsError):
            export_patient_reports(self.review, self.tiles, self.manifest, output)
        self.assertEqual(before, (output / "summary.json").read_bytes())

    @unittest.skipUnless(find_spec("reportlab"), "未安装可选 PDF 依赖")
    def test_render_failure_does_not_publish_partial_reports(self) -> None:
        """排版失败不发布一半患者文件，也不留下临时报告目录。"""
        output = self.root / "out"
        with patch("leukocyte_analysis.evaluation.patient_pdf.resolve_font", return_value=Path("font.ttf")):
            with patch("leukocyte_analysis.evaluation.patient_pdf.render_patient_pdf",
                       side_effect=RuntimeError("simulated")):
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    export_patient_reports(self.review, self.tiles, self.manifest, output)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob(".report-*")))

    def test_cli_only_delegates(self) -> None:
        """CLI 不启动模型，路径参数原样交给报告导出业务。"""
        argv = ["leukocyte-patient-report", "--review-dir", "review", "--tiles-npy", "tiles.npy",
                "--patient-manifest", "patients.csv", "--output-dir", "out"]
        with patch.object(sys, "argv", argv):
            self.assertEqual(parse_args().patient_manifest, Path("patients.csv"))
            with patch("leukocyte_analysis.patient_report.export_patient_reports",
                       return_value={"patients": []}) as export, patch("builtins.print"):
                main()
        export.assert_called_once_with(Path("review"), Path("tiles.npy"), Path("patients.csv"),
                                       Path("out"), font_ttf=None)
