"""独立组合质量与 RMD 特征偏离规则，不参与模型训练。"""

import math
from dataclasses import dataclass

from ..data.bbbc045_config import CLASS_NAMES


@dataclass(frozen=True)
class ReviewPolicy:
    """RMD 上限与清晰度下限分开配置；未指定模糊阈值时不自动判废。"""

    ood_threshold: float
    focus_min: float | None = None
    focus_review_min: float | None = None

    def __post_init__(self) -> None:
        """拒绝 NaN、负清晰度阈值及互相矛盾的严重／边界阈值。"""
        if not math.isfinite(self.ood_threshold):
            raise ValueError("RMD 阈值必须有限，可为负数")
        for value in (self.focus_min, self.focus_review_min):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("清晰度阈值必须是非负有限数")
        if (self.focus_min is not None and self.focus_review_min is not None
                and self.focus_min > self.focus_review_min):
            raise ValueError("严重模糊下限不能高于质量存疑下限")

    def assess(self, predicted: int, ood_score: float,
               quality: dict[str, float]) -> dict:
        """只读取模型评分和图片质量；真实标签不影响实际处理决定。"""
        if (predicted not in range(len(CLASS_NAMES)) or not math.isfinite(ood_score)
                or any(not math.isfinite(value) or value < 0 for value in quality.values())):
            raise ValueError("预测类别、特征分数或清晰度评分无效")
        focus = quality["normalized_tenengrad"]
        reasons = []
        quality_status = "not_assessed"
        if self.focus_min is not None or self.focus_review_min is not None:
            quality_status = "passed"
        # 常数图没有可供分类的灰度结构；与尚未核定的模糊阈值分开记录。
        if quality["contrast_std"] == 0:
            quality_status = "unusable"
            reasons.append("no_contrast")
        elif self.focus_min is not None and focus < self.focus_min:
            quality_status = "unusable"
            reasons.append("severe_blur_candidate")
        elif self.focus_review_min is not None and focus < self.focus_review_min:
            quality_status = "needs_review"
            reasons.append("low_focus")
        if ood_score > self.ood_threshold:
            reasons.append("feature_outlier")
        decision = ("unusable" if quality_status == "unusable" else
                    "needs_review" if reasons else "accepted")
        return {"decision": decision,
                "decision_label": CLASS_NAMES[predicted] if decision == "accepted" else
                                  "unusable" if decision == "unusable" else "uncertain",
                "quality_status": quality_status, "review_reason": ";".join(reasons)}
