"""患者报告的纯数据模型与样本统计；不读取文件、不执行推理。"""

from collections import Counter
from dataclasses import dataclass
import math

from ..data.bbbc045_config import CLASS_NAMES


CLASS_TITLES = ("淋巴细胞", "嗜酸性粒细胞", "单核细胞", "中性粒细胞")
# 仅控制报告顺序，不改变分类头和历史标签编号。
REPORT_CLASS_ORDER = (3, 0, 2, 1)
DECISIONS = ("accepted", "needs_review", "unusable")
REASON_TITLES = {
    "feature_outlier": "特征偏离",
    "no_contrast": "缺少灰度结构",
    "severe_blur_candidate": "低于清晰度下限",
    "low_focus": "清晰度存疑",
}


def checked_identifier(value: str, name: str) -> str:
    """限制标识长度与控制字符；不把未经处理的标识用作文件路径。"""
    if (not isinstance(value, str) or not value.strip() or len(value) > 64
            or any(not char.isprintable() for char in value)):
        raise ValueError(f"{name} 须为 1-64 个可显示字符")
    return value.strip()


@dataclass(frozen=True)
class CellResult:
    """一张原始图块的已生成预测与明确的患者／样本归属。"""

    packed_index: int
    patient_id: str
    sample_id: str
    cell_id: str
    predicted_index: int
    decision: str
    confidence: float
    ood_score: float
    focus: float
    quality_status: str
    review_reason: str

    def __post_init__(self) -> None:
        """拒绝非法计数、非有限评分以及无法解释的决策状态。"""
        for name in ("patient_id", "sample_id", "cell_id"):
            if checked_identifier(getattr(self, name), name) != getattr(self, name):
                raise ValueError(f"{name} 不应包含首尾空白")
        if (self.packed_index < 0 or self.predicted_index not in range(len(CLASS_NAMES))
                or self.decision not in DECISIONS
                or not all(math.isfinite(x) for x in
                           (self.confidence, self.ood_score, self.focus))
                or not 0 <= self.confidence <= 1 or self.focus < 0):
            raise ValueError("图块索引、类别、状态或评分无效")
        reasons = self.review_reason.split(";") if self.review_reason else []
        if (any(reason not in REASON_TITLES for reason in reasons)
                or len(set(reasons)) != len(reasons)
                or (self.decision == "accepted") != (not reasons)):
            raise ValueError("复核原因与决策状态不一致")
        if self.quality_status not in {"not_assessed", "passed", "needs_review", "unusable"}:
            raise ValueError("图片质量状态无效")


def group_patients(cells: list[CellResult]) -> dict[str, dict[str, list[CellResult]]]:
    """同一患者合为一份报告，但每个样本保留独立分组与分母。"""
    if not cells:
        raise ValueError("患者清单不能为空")
    patients, owners, indices, identities = {}, {}, set(), set()
    for cell in sorted(cells, key=lambda item: item.packed_index):
        identity = (cell.patient_id, cell.sample_id, cell.cell_id)
        if cell.packed_index in indices or identity in identities:
            raise ValueError("图块索引或样本内 cell_id 重复")
        if owners.setdefault(cell.sample_id, cell.patient_id) != cell.patient_id:
            raise ValueError("同一 sample_id 不能分配给不同患者")
        indices.add(cell.packed_index)
        identities.add(identity)
        patients.setdefault(cell.patient_id, {}).setdefault(cell.sample_id, []).append(cell)
    return patients


def summarize_sample(cells: list[CellResult]) -> dict:
    """保留 accepted 内部比例；另给出含复核／不可用项的全输入总表比例。"""
    if not cells or len({(c.patient_id, c.sample_id) for c in cells}) != 1:
        raise ValueError("一次只能统计一个非空样本")
    counts = Counter(cell.decision for cell in cells)
    classes = Counter(c.predicted_index for c in cells if c.decision == "accepted")
    accepted = counts["accepted"]
    report_rows = [{"key": CLASS_NAMES[index], "title": CLASS_TITLES[index],
                    "count": classes[index], "proportion": classes[index] / len(cells)}
                   for index in REPORT_CLASS_ORDER]
    report_rows.extend({"key": key, "title": title, "count": counts[key],
                        "proportion": counts[key] / len(cells)}
                       for key, title in (("needs_review", "待复核"), ("unusable", "质量不合格")))
    return {
        "patient_id": cells[0].patient_id, "sample_id": cells[0].sample_id,
        "total": len(cells), "decisions": {key: counts[key] for key in DECISIONS},
        "quality_not_assessed": sum(c.quality_status == "not_assessed" for c in cells),
        "report_rows": report_rows,
        "classes": [{"class": name, "title": CLASS_TITLES[index],
                     "count": classes[index],
                     "proportion": classes[index] / accepted if accepted else None}
                    for index, name in enumerate(CLASS_NAMES)],
    }


def report_groups(cells: list[CellResult]) -> list[tuple[dict, list[CellResult]]]:
    """使总表、分类栏与图册顺序共用同一组互斥分类，避免数量与图片错位。"""
    groups = {key: [] for key in (*CLASS_NAMES, "needs_review", "unusable")}
    for cell in cells:
        key = CLASS_NAMES[cell.predicted_index] if cell.decision == "accepted" else cell.decision
        groups[key].append(cell)
    return [(row, sorted(groups[row["key"]], key=lambda cell: cell.packed_index))
            for row in summarize_sample(cells)["report_rows"]]
