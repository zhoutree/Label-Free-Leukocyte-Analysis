"""非破坏性的图块质量代理指标；不据此自动删除细胞。"""

import numpy as np
from scipy.ndimage import gaussian_filter, laplace


def focus_proxy(tile: np.ndarray) -> float:
    """返回中央区域平滑后拉普拉斯响应的方差。

    低分只提示缺少高频细节；该代理值不能单独判定模糊或错标。
    """
    if tile.ndim != 2 or min(tile.shape) < 10 or not np.isfinite(tile).all():
        raise ValueError("清晰度代理指标需要有限像素的二维图块")

    # 只看中央区域，减轻拼图边界和远处背景的影响。
    margin = min(tile.shape) // 10
    central = np.asarray(tile[margin:-margin, margin:-margin], dtype=np.float64)
    smoothed = gaussian_filter(central, sigma=0.8)
    return float(np.var(laplace(smoothed)))
