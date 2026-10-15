"""验证四分类标签、受试者划分和图块读取的约定。"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from leukocyte_analysis.data.bbbc045_config import CLASS_NAMES, model_label
from leukocyte_analysis.data.dataset import LeukocyteDataset
from leukocyte_analysis.data.split import split_for_donor, summarize_split_groups
from leukocyte_analysis.prepare_data import prepare_data


class TestDataRules(unittest.TestCase):
    """模型标签合并不改变原始清单，受试者年份不跨数据划分。"""

    def test_b_and_t_map_to_lymphocyte(self) -> None:
        """B/T 只在模型标签层合并，未知原始类别必须报错。"""
        self.assertEqual(CLASS_NAMES,
                         ("lymphocyte", "eosinophil", "monocyte", "neutrophil"))
        self.assertEqual(model_label("B"), "lymphocyte")
        self.assertEqual(model_label("T"), "lymphocyte")
        with self.assertRaises(ValueError):
            model_label("unknown")

    def test_crf101_years_stay_together_in_regular_training(self) -> None:
        """常规划分也不能把 CRF101 两个年份分到不同集合。"""
        self.assertEqual(split_for_donor("CRF101_2014"), "train")
        self.assertEqual(split_for_donor("CRF101_2015"), "train")
        records = [
            {"donor": "CRF101_2014", "split": "train"},
            {"donor": "CRF101_2015", "split": "train"},
            {"donor": "CRF049", "split": "test"},
        ]
        groups = summarize_split_groups(records)
        self.assertIn({"group": "CRF101", "split": "train",
                       "donors": ["CRF101_2014", "CRF101_2015"], "tiles": 2}, groups)
        records[1]["split"] = "val"
        with self.assertRaisesRegex(ValueError, "CRF101"):
            summarize_split_groups(records)
        records[1]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "CRF101"):
            summarize_split_groups(records)


class TestLeukocyteDataset(unittest.TestCase):
    """检查图块索引、B/T 映射以及增强与归一化的先后顺序。"""

    def test_dataset_maps_b_and_t_without_rewriting_manifest(self) -> None:
        """读取图块时映射 B/T，仍保留 CSV 原始类别和行索引。"""
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            split_csv, tiles_npy, stats_json = (root / "split.csv", root / "tiles.npy",
                                                root / "stats.json")
            with split_csv.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=("split", "label", "donor"))
                writer.writeheader()
                # 保留原始 B/T；只在数据集读取时映射为淋巴细胞。
                for label in ("B", "T", "eosinophil", "neutrophil"):
                    writer.writerow({"split": "train", "label": label, "donor": "synthetic"})
            np.save(tiles_npy, np.zeros((4, 55, 55), dtype=np.float32))
            stats_json.write_text(json.dumps({"mean": 0, "std": 1}), encoding="utf-8")

            dataset = LeukocyteDataset(split_csv, tiles_npy, stats_json, "train")
            self.assertEqual([row[1] for row in dataset.samples], [0, 0, 1, 3])
            self.assertEqual(tuple(dataset[0][0].shape), (1, 55, 55))
            indexed = LeukocyteDataset(split_csv, tiles_npy, stats_json, "train",
                                       include_index=True)
            self.assertEqual(indexed[2][1:], (1, 2))
            # Windows 上需先释放内存映射，再清理临时 .npy。
            del dataset, indexed

    def test_raw_transform_precedes_normalization(self) -> None:
        """原灰度扰动先于训练统计量标准化，验证图块保持原样。"""
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            split_csv, tiles_npy, stats_json = (root / "split.csv", root / "tiles.npy",
                                                root / "stats.json")
            with split_csv.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=("split", "label", "donor"))
                writer.writeheader()
                writer.writerow({"split": "train", "label": "B", "donor": "synthetic"})
                writer.writerow({"split": "val", "label": "B", "donor": "synthetic2"})
            np.save(tiles_npy, np.full((2, 55, 55), 2, dtype=np.float32))
            stats_json.write_text(json.dumps({"mean": 2, "std": 2}), encoding="utf-8")

            train = LeukocyteDataset(split_csv, tiles_npy, stats_json, "train",
                                     raw_transform=lambda image: image + 2)
            val = LeukocyteDataset(split_csv, tiles_npy, stats_json, "val")
            self.assertEqual(float(train[0][0].mean()), 1.0)
            self.assertEqual(float(val[0][0].mean()), 0.0)
            self.assertEqual(tuple(train[0][0].shape), (1, 55, 55))
            del train, val


class TestDataPreparation(unittest.TestCase):
    """用少量合成 TIFF 核对准备入口的顺序、输出和防覆盖规则。"""

    def test_prepare_data_uses_existing_pipeline(self) -> None:
        """原始标签与受试者划分保留，统计量仅来自训练图块。"""
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            data_root, output_dir = root / "raw", root / "processed"
            train_tile = np.arange(55 * 55, dtype=np.float32).reshape(55, 55) / 100000
            for donor, label, tile in (
                ("CRF022", "T", train_tile),
                ("CRF034", "B", np.full((55, 55), 2, dtype=np.float32)),
                ("CRF049", "monocyte", np.full((55, 55), 3, dtype=np.float32)),
            ):
                source = data_root / donor / label / "batch1" / "ch5_1.tif"
                source.parent.mkdir(parents=True)
                tifffile.imwrite(source, tile)

            prepare_data(data_root, output_dir)
            with (output_dir / "bbbc045_split.csv").open(newline="", encoding="utf-8") as file:
                records = list(csv.DictReader(file))
            self.assertEqual([(row["label"], row["split"]) for row in records],
                             [("T", "train"), ("B", "val"), ("monocyte", "test")])
            tiles = np.load(output_dir / "bbbc045_tiles.npy", mmap_mode="r")
            self.assertEqual(tiles.shape, (3, 55, 55))
            np.testing.assert_array_equal(tiles[0], train_tile)
            stats = json.loads((output_dir / "bbbc045_train_stats.json").read_text("utf-8"))
            self.assertEqual(stats["train_tiles"], 1)
            self.assertAlmostEqual(stats["mean"], float(train_tile.mean()), places=6)
            del tiles

            with self.assertRaises(FileExistsError):
                prepare_data(data_root, output_dir)
