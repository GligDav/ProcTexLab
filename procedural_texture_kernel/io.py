"""Raster loading and scalar normalization."""
from pathlib import Path
import numpy as np
from PIL import Image

def normalize_image(image: np.ndarray) -> np.ndarray:
    """Convert grayscale/RGB/RGBA input to finite float64 scalar values in [0, 1]."""
    arr = np.asarray(image)
    if arr.size == 0 or arr.ndim not in (2, 3):
        raise ValueError("image must be a non-empty 2D grayscale or 3D RGB/RGBA array")
    if arr.ndim == 3:
        if arr.shape[2] not in (3, 4):
            raise ValueError("color images must have 3 or 4 channels")
        arr = arr[..., :3].astype(np.float64)
        arr = arr @ np.array([0.2126, 0.7152, 0.0722])
    elif arr.dtype.kind in "ui":
        maximum = np.iinfo(arr.dtype).max
        arr = arr.astype(np.float64) / maximum
        return arr
    else:
        arr = arr.astype(np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("image contains NaN or infinite values")
    if image.dtype.kind in "ui":
        arr /= np.iinfo(image.dtype).max
    elif arr.min() < 0 or arr.max() > 1:
        lo, hi = float(arr.min()), float(arr.max())
        arr = np.zeros_like(arr) if hi == lo else (arr - lo) / (hi - lo)
    return np.clip(arr, 0, 1)

def load_image(path: str | Path) -> np.ndarray:
    """Load a common Pillow-supported raster and return normalized grayscale."""
    try:
        with Image.open(path) as image:
            return normalize_image(np.asarray(image))
    except (OSError, ValueError) as exc:
        raise ValueError(f"failed to load image {path!s}: {exc}") from exc
