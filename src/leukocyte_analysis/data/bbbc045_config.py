"""BBBC045 数据格式、类别及受试者划分的统一设定。"""

TILE_SIZE = 55
BRIGHTFIELD_GLOB = "CRF*/*/batch*/ch5_*.tif"

# 顺序决定模型输出的类别编号，已有数据使用后不要随意调整。
CLASS_NAMES = ("B", "T", "eosinophil", "monocyte", "neutrophil")

TRAIN_DONORS = frozenset({
    "CRF022", "CRF041", "CRF074", "CRF101_2014",
    "CRF101_2015", "CRF130", "CRF132", "CRF186", "CRF59",
})
VAL_DONORS = frozenset({"CRF034", "CRF102"})
TEST_DONORS = frozenset({"CRF049", "CRF066"})