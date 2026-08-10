"""Canonical normalized image coordinates."""
from functools import lru_cache
import math
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

def coordinate_grid_region(
    width: int, height: int, u_bounds: tuple[float, float], v_bounds: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Return a grid sampling an arbitrary half-open rectangular UV region."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    u_min, u_max = map(float, u_bounds)
    v_min, v_max = map(float, v_bounds)
    if not all(map(math.isfinite, (u_min, u_max, v_min, v_max))):
        raise ValueError("UV bounds must be finite")
    if u_max <= u_min or v_max <= v_min:
        raise ValueError("each UV upper bound must be greater than its lower bound")
    u = u_min + (u_max - u_min) * np.arange(width, dtype=np.float64) / width
    v = v_min + (v_max - v_min) * np.arange(height, dtype=np.float64) / height
    return np.meshgrid(u, v)
