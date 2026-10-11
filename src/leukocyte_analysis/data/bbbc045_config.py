"""BBBC045 数据格式、类别及受试者划分的统一设定。"""

TILE_SIZE = 55
BRIGHTFIELD_GLOB = "CRF*/*/batch*/ch5_*.tif"

SOURCE_CLASS_NAMES = ("B", "T", "eosinophil", "monocyte", "neutrophil")
# 模型的四个输出按此顺序编号；修改顺序会使已训练权重失效。
CLASS_NAMES = ("lymphocyte", "eosinophil", "monocyte", "neutrophil")


def model_label(raw_label: str) -> str:
    """读取时合并 B/T，原始清单仍保留可追溯的五类标签。"""
    if raw_label not in SOURCE_CLASS_NAMES:
        raise ValueError(f"未知原始类别：{raw_label}")
    if raw_label in {"B", "T"}:
        return "lymphocyte"
    return raw_label

TRAIN_DONORS = frozenset({
    "CRF022", "CRF041", "CRF074", "CRF101_2014",
    "CRF101_2015", "CRF130", "CRF132", "CRF186", "CRF59",
})
VAL_DONORS = frozenset({"CRF034", "CRF102"})
TEST_DONORS = frozenset({"CRF049", "CRF066"})
