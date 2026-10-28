"""验证错分导出业务及薄命令入口，不读取真实数据或检查点。"""

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from leukocyte_analysis.data.bbbc045_config import CLASS_NAMES
from leukocyte_analysis.evaluation.error_export import export_errors
from leukocyte_analysis.record_errors import main


class TestErrorExport(unittest.TestCase):
    """业务 API 独立保护测试集、防覆盖和完整预测顺序。"""

    def setUp(self) -> None:
        """准备一对可追溯预测，其中单核细胞被错分为中性粒细胞。"""
        self.dataset = SimpleNamespace(
            samples=[(0, 0, "CRF034"), (1, 2, "CRF034")], class_names=CLASS_NAMES,
            source_by_index={index: {"file": f"CRF034/{label}/batch1/ch5_1.tif",
                                     "donor": "CRF034", "source_label": label,
                                     "row": 0, "col": index}
                             for index, label in ((0, "B"), (1, "monocyte"))})
        self.rows = [{"packed_index": 0, "true_index": 0, "predicted_index": 0,
                      "confidence": 0.9},
                     {"packed_index": 1, "true_index": 2, "predicted_index": 3,
                      "confidence": 0.8}]

    def test_existing_predictions_export_without_model_and_reject_reordered_rows(self) -> None:
        """测试集仅复用现有预测，核对完整顺序，失败不发布错分表。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "predictions.csv"
            with patch("leukocyte_analysis.evaluation.error_export.LeukocyteDataset",
                       return_value=self.dataset), patch(
                       "leukocyte_analysis.evaluation.error_export.predict") as predict:
                for number, rows in enumerate((self.rows, self.rows[::-1])):
                    with source.open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
                    output = root / f"errors_{number}.csv"
                    args = (Path("split.csv"), Path("tiles.npy"), Path("stats.json"), "test", output)
                    if number:
                        with self.assertRaisesRegex(ValueError, "顺序不一致"):
                            export_errors(*args, predictions_csv=source)
                        self.assertFalse(output.exists())
                    else:
                        self.assertEqual(export_errors(*args, predictions_csv=source), 1)
                        with output.open(newline="", encoding="utf-8") as handle:
                            self.assertEqual(next(csv.DictReader(handle))["true_label"], "monocyte")
                        original = output.read_bytes()
                        with self.assertRaises(FileExistsError):
                            export_errors(*args, predictions_csv=source)
                        self.assertEqual(output.read_bytes(), original)
                predict.assert_not_called()

    def test_validation_checkpoint_uses_shared_inference(self) -> None:
        """验证检查点只调用共享恢复和推理函数，不实现另一套前向循环。"""
        module = "leukocyte_analysis.evaluation.error_export"
        with tempfile.TemporaryDirectory() as temporary:
            with (patch(f"{module}.LeukocyteDataset", return_value=self.dataset),
                  patch(f"{module}.check_checkpoint_head") as check,
                  patch(f"{module}.build_resnet18"), patch(f"{module}.load_classifier"),
                  patch(f"{module}.predict", return_value=self.rows) as predict):
                count = export_errors(Path("split"), Path("tiles"), Path("stats"), "val",
                                      Path(temporary) / "errors.csv", checkpoint=Path("best.ckpt"))
                self.assertEqual(count, 1)
                check.assert_called_once_with(Path("best.ckpt"))
                predict.assert_called_once()

    def test_invalid_direct_calls_are_rejected_before_loading_data(self) -> None:
        """绕过 CLI 也不能混用来源或对测试集重新推理。"""
        base = (Path("split"), Path("tiles"), Path("stats"))
        variants = [("val", {}), ("test", {"checkpoint": Path("best")}),
                    ("train", {"predictions_csv": Path("pred")}),
                    ("val", {"predictions_csv": Path("pred"), "batch_size": 0}),
                    ("val", {"predictions_csv": Path("pred"), "checkpoint": Path("best")})]
        with patch("leukocyte_analysis.evaluation.error_export.LeukocyteDataset") as dataset:
            for split, options in variants:
                with self.subTest(split=split, options=options), self.assertRaises(ValueError):
                    export_errors(*base, split, Path("unused.csv"), **options)
            dataset.assert_not_called()

    def test_cli_delegates_business_to_export_function(self) -> None:
        """命令入口仅传参并显示数量，业务不在 main 中重复实现。"""
        argv = ["record", "--split", "test", "--predictions-csv", "pred.csv",
                "--output-csv", "errors.csv"]
        with patch.object(sys, "argv", argv), patch(
                "leukocyte_analysis.record_errors.export_errors", return_value=1) as export:
            main()
        self.assertEqual(export.call_args.args[3:], ("test", Path("errors.csv")))
        self.assertEqual(export.call_args.kwargs["predictions_csv"], Path("pred.csv"))
