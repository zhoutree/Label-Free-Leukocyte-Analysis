"""非破坏性的图块质量代理指标；不据此自动删除细胞。"""

import numpy as np
from scipy.ndimage import gaussian_filter, laplace, sobel


def _focus_region(tile: np.ndarray) -> np.ndarray:
    """校验原始灰度并取中央区域，避免拼图边界干扰清晰度评分。"""
    if tile.ndim != 2 or min(tile.shape) < 10 or not np.isfinite(tile).all():
        raise ValueError("清晰度代理指标需要有限像素的二维图块")
    margin = min(tile.shape) // 10
    return np.asarray(tile[margin:-margin, margin:-margin], dtype=np.float64)


def focus_proxy(tile: np.ndarray) -> float:
    """返回中央区域平滑后拉普拉斯响应的方差。

    低分只提示缺少高频细节；该代理值不能单独判定模糊或错标。
    """
    central = _focus_region(tile)
    smoothed = gaussian_filter(central, sigma=0.8)
    return float(np.var(laplace(smoothed)))


def focus_measurements(tile: np.ndarray) -> dict[str, float]:
    """返回原始拉普拉斯、Sobel 梯度能量和对比度归一化梯度。

    归一化项减轻纯亮度偏移、线性对比度缩放的影响；不表示可用概率，
    也不能排除噪声、低纹理及非线性成像差异的影响。模型输入不被修改。
    """
    central = _focus_region(tile)
    smoothed = gaussian_filter(central, sigma=0.8)
    variance = float(np.var(central)) if np.ptp(central) > 0 else 0.0
    energy = float(np.mean(sobel(smoothed, axis=0) ** 2 + sobel(smoothed, axis=1) ** 2))
    result = {
        "laplacian_variance": float(np.var(laplace(smoothed))),
        "tenengrad": energy,
        "contrast_std": float(np.sqrt(variance)),
        "normalized_tenengrad": energy / variance if variance > 0 else 0.0,
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("灰度动态范围异常，无法计算有限的清晰度评分")
    return result
