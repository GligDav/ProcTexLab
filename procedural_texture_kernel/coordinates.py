"""Canonical normalized image coordinates."""
from functools import lru_cache
import numpy as np

COORDINATE_SYSTEM = "uv_normalized_top_left_v_down_half_open"

@lru_cache(maxsize=32)
def coordinate_grid(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(u, v)`` grids of shape ``(height, width)`` on ``[0, 1)``.

    The origin is the top-left pixel, U increases right, and V increases down.
    Angles are counter-clockwise in this image-coordinate plane (therefore they
    look clockwise in a conventional Y-up plot).
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    u = np.arange(width, dtype=np.float64) / width
    v = np.arange(height, dtype=np.float64) / height
    return np.meshgrid(u, v)
