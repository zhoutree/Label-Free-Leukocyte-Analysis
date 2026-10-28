"""封装 PyTorch-OOD 的 RMD；仅拟合已知类特征，不更新分类网络。"""

from importlib.metadata import version

import numpy as np
import torch
from torch import nn

from ..data.bbbc045_config import CLASS_NAMES


RMD_STATE_FIELDS = ("mu", "cov", "precision", "background_mu", "background_cov",
                    "background_precision")


def make_detector():
    """延迟导入可选依赖，让原训练和 CSV 复核功能保持可独立使用。"""
    try:
        from pytorch_ood.detector import RMD
    except ImportError as error:
        raise ImportError('特征拒识需要先安装 python -m pip install -e ".[review]"') from error
    if version("pytorch-ood") != "0.4.0":
        raise ValueError("RMD 状态接口已验证版本为 pytorch-ood==0.4.0，请使用此版本")
    return RMD(encoder=None)


class ResNetFeatures(nn.Module):
    """共享原 ResNet 的模块，一次前向同时取得池化特征和原始 logits。"""

    def __init__(self, backbone: nn.Module) -> None:
        """复用分类头前的全部层；不替换参数、不增加可训练层。"""
        super().__init__()
        self.body = nn.Sequential(*list(backbone.children())[:-1])
        self.head = backbone.fc

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回原四分类 logits 及尚未做 softmax 的 512 维描述。"""
        features = self.body(images).flatten(1)
        return self.head(features), features


def fit_detector(features: torch.Tensor, labels: torch.Tensor):
    """只用训练集建立四类和背景分布，沿用第三方实现的 float32 状态。"""
    if (features.ndim != 2 or features.shape[1] < 1 or labels.ndim != 1
            or len(features) != len(labels) or not torch.isfinite(features).all()):
        raise ValueError("RMD 需要有限的二维特征及一一对应的标签")
    if set(labels.tolist()) != set(range(len(CLASS_NAMES))):
        raise ValueError("RMD 参考集必须包含全部四类，不能把已训练类别伪装成未知类")
    if any(int((labels == index).sum()) < 2 for index in range(len(CLASS_NAMES))):
        raise ValueError("RMD 每个类别至少需要两个训练样本")
    detector = make_detector()
    detector.fit_features(features.detach().cpu().float(), labels.detach().cpu().long())
    detector_state(detector)
    return detector


def detector_state(detector) -> dict[str, torch.Tensor]:
    """只导出有限张量；保存统计量而非整个 Python 检测器对象。"""
    state = {name: getattr(detector, name).detach().cpu() for name in RMD_STATE_FIELDS}
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError("RMD 拟合出现非有限统计量")
    return state


def restore_detector(state: dict, dimension: int):
    """核对四类中心和协方差规格，恢复已拟合状态。"""
    shapes = {"mu": (len(CLASS_NAMES), dimension), "background_mu": (dimension,)}
    if dimension < 1 or set(state) != set(RMD_STATE_FIELDS):
        raise ValueError("RMD 状态字段不完整")
    detector = make_detector()
    for name, value in state.items():
        expected = shapes.get(name, (dimension, dimension))
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != expected
                or not value.is_floating_point() or not torch.isfinite(value).all()):
            raise ValueError(f"RMD 统计量规格异常：{name}")
        setattr(detector, name, value.cpu().float())
    return detector


def outlier_scores(detector, features: torch.Tensor) -> np.ndarray:
    """返回越大越偏离的 RMD 分数；分数可为负数，不是概率。"""
    scores = detector.predict_features(features.detach().cpu().float()).numpy()
    if scores.shape != (len(features),) or not np.isfinite(scores).all():
        raise ValueError("RMD 输出分数无效")
    return scores


def known_threshold(scores: np.ndarray, reject_rate: float) -> float:
    """按完整已知类验证分数选择上分位数，并列值保留，实测误拒可更低。"""
    scores = np.asarray(scores, dtype=np.float64)
    if (scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all()
            or not np.isfinite(reject_rate) or not 0 < reject_rate < 1):
        raise ValueError("已知类误拒目标须在 (0, 1)，验证分数须非空且有限")
    return float(np.quantile(scores, 1 - reject_rate, method="higher"))
