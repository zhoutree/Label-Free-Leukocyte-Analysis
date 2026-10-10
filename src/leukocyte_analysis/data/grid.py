"""将白细胞 montage 按固定网格裁成单细胞候选图块，并跳过全零格子。

这里只做网格裁切，不做细胞边缘分割或类别判断。
"""

from collections.abc import Iterator

import numpy as np

from .bbbc045_config import TILE_SIZE


def iter_nonempty_tiles(
    image: np.ndarray, tile_size: int = TILE_SIZE
) -> Iterator[tuple[int, int, np.ndarray]]:
    """依次返回非空格的 (行号, 列号, 55×55 原始像素)。"""
    if image.ndim != 2:
        raise ValueError("只接受单通道二维图像")

    height, width = image.shape
    if height % tile_size or width % tile_size:
        raise ValueError("图像尺寸必须是格子尺寸的整数倍")

    for row in range(height // tile_size):
        for col in range(width // tile_size):
            y, x = row * tile_size, col * tile_size
            tile = image[y:y + tile_size, x:x + tile_size]
            if np.any(tile != 0):
                yield row, col, tile