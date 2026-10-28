"""验证共享报告文件工具的拒绝覆盖、更新和失败清理，不加载模型。"""

import csv
import tempfile
import unittest
from pathlib import Path

from leukocyte_analysis.reporting import new_report, write_csv


class TestReportIO(unittest.TestCase):
    """不同业务共用工具后仍保留新建预测与更新频次表的区别。"""

    def test_csv_creation_preserves_fields_and_refuses_overwrite(self) -> None:
        """按明确列顺序保存中文及特殊字符，再次创建不破坏已有内容。"""
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            path = Path(temporary) / "predictions.csv"
            row = {"label": "淋巴,细胞\n候选", "index": 17}
            write_csv(path, [row], ("index", "label"), overwrite=False)
            original = path.read_bytes()
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, ["index", "label"])
                self.assertEqual(list(reader), [{"index": "17", "label": row["label"]}])
            with self.assertRaises(FileExistsError):
                write_csv(path, [], ("index", "label"), overwrite=False)
            self.assertEqual(path.read_bytes(), original)

    def test_csv_update_and_empty_header(self) -> None:
        """允许更新的累计表可以原子替换；无记录时仍保留表头。"""
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            path = Path(temporary) / "frequency.csv"
            write_csv(path, [{"count": 1}], ("count",), overwrite=False)
            write_csv(path, [{"count": 2}], ("count",), overwrite=True)
            self.assertEqual(path.read_text(encoding="utf-8"), "count\n2\n")
            write_csv(path, [], ("count",), overwrite=True)
            self.assertEqual(path.read_text(encoding="utf-8"), "count\n")

    def test_failed_csv_update_keeps_original_and_removes_temporary_file(self) -> None:
        """字段校验失败时，旧累计表原样保留且不留下暂存文件。"""
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            path = root / "frequency.csv"
            write_csv(path, [{"count": 1}], ("count",))
            original = path.read_bytes()
            with self.assertRaises(ValueError):
                write_csv(path, [{"unexpected": 2}], ("count",))
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(root.iterdir()), [path])

    def test_report_publishes_only_on_success_and_refuses_existing_directory(self) -> None:
        """失败不发布半成品；成功发布后再次申请同一目录必须失败。"""
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            output = root / "report"
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                with new_report(output) as stage:
                    (stage / "partial.txt").write_text("partial", encoding="utf-8")
                    raise RuntimeError("simulated")
            self.assertEqual(list(root.iterdir()), [])
            with new_report(output) as stage:
                (stage / "complete.txt").write_text("complete", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                with new_report(output):
                    self.fail("已有报告不应进入暂存上下文")
            self.assertEqual((output / "complete.txt").read_text(encoding="utf-8"), "complete")
