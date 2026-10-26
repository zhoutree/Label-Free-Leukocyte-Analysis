"""检查运行记录路径可迁移，并兼容旧版绝对路径。"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from leukocyte_analysis.training.run_paths import resolve_run_path, store_run_path
from leukocyte_analysis.data.cv_split import DonorFold
from leukocyte_analysis.training.cv_plan import prepare_cv_run
from leukocyte_analysis.training.cv_reports import fold_metrics


class TestRunPaths(unittest.TestCase):
    """不访问正式数据文件，只核对路径解析规则。"""

    def test_run_relative_path_for_same_and_child_directory(self) -> None:
        """运行目录自身及子目录应保留原有的相对表示。"""
        run_dir = Path(__file__).resolve().parent / "unused_run"
        self.assertEqual(store_run_path(run_dir, run_dir), ".")
        self.assertEqual(store_run_path(run_dir / "best.ckpt", run_dir), "best.ckpt")

    def test_moving_project_preserves_run_relative_path(self) -> None:
        """数据和运行目录一起迁移后，计划仍指向新位置。"""
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            stored = store_run_path(root / "old" / "data" / "tiles.npy",
                                    root / "old" / "runs" / "folds")
            self.assertFalse(Path(stored).is_absolute())
            self.assertNotIn("\\", stored)
            self.assertEqual(resolve_run_path(root / "new" / "runs" / "folds", stored),
                             root / "new" / "data" / "tiles.npy")

    def test_override_replaces_legacy_absolute_path(self) -> None:
        """旧计划路径失效时允许显式指定当前数据位置。"""
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            old = str(root / "old" / "tiles.npy")
            replacement = root / "new" / "tiles.npy"
            self.assertEqual(resolve_run_path(root / "run", old), Path(old))
            self.assertEqual(resolve_run_path(root / "run", old, replacement), replacement)

    def test_old_plan_can_resume_with_relocated_inputs_and_checkpoint(self) -> None:
        """旧绝对路径失效时，仅显式迁移输入可复用已完成折。"""
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            run_dir = Path(temporary) / "run"
            fold_dir = run_dir / "fold_01"
            (fold_dir / "training" / "checkpoints").mkdir(parents=True)
            (fold_dir / "training" / "checkpoints" / "best.ckpt").touch()
            rows = [{"packed_index": 0, "donor": "A", "true_index": 0,
                     "predicted_index": 0}]
            with (fold_dir / "validation_predictions.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            saved = {"source_manifest": "C:/old/split.csv", "packed_tiles": "C:/old/tiles.npy",
                     "training": {"label_smoothing": 0.0}}
            current = {**saved, "source_manifest": "../../data/split.csv",
                       "packed_tiles": "../../data/tiles.npy"}
            fold = DonorFold(1, frozenset({"B"}), frozenset({"A"}))
            result = {"fold": 1, "val_tiles": 1, "metrics": fold_metrics(rows),
                      "checkpoint": "C:/old/run/fold_01/training/checkpoints/best.ckpt"}
            (run_dir / "plan.json").write_text(json.dumps(saved), encoding="utf-8")
            (run_dir / "summary.json").write_text(json.dumps({
                "partial": True, "plan": saved, "fold_results": [result],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "原运行的分折"):
                prepare_cv_run(run_dir, current.copy(), [fold], continue_run=True,
                               label_smoothing=0.0)
            results, predictions = prepare_cv_run(
                run_dir, current.copy(), [fold], continue_run=True,
                label_smoothing=0.0, allow_relocated_inputs=True,
            )
            self.assertEqual(len(predictions), 1)
            self.assertEqual(resolve_run_path(run_dir, results[0]["checkpoint"]),
                             fold_dir / "training" / "checkpoints" / "best.ckpt")


if __name__ == "__main__":
    unittest.main()
