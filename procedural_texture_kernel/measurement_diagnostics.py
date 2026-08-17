"""Absolute spatial and directional diagnostics for texture comparisons."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


DEFAULT_CONTRAST_SCALES = (1.0, 2.0, 4.0, 8.0)


def _images(target: np.ndarray, result: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first = np.asarray(target, dtype=np.float64)
    second = np.asarray(result, dtype=np.float64)
    if (first.ndim != 2 or first.size == 0 or not np.all(np.isfinite(first))
            or not np.all(np.isfinite(second))):
        raise ValueError("target and result must be finite, non-empty 2D images")
    if first.shape != second.shape:
        raise ValueError("target and result must have matching shapes")
    return first, second


def _ratio(target: float, result: float) -> float:
    epsilon = np.finfo(float).eps * max(abs(target), abs(result), 1.0)
    if abs(target) <= epsilon and abs(result) <= epsilon:
        return 1.0
    return float(result / max(target, epsilon))


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    dy = (np.gradient(image, axis=0) if image.shape[0] > 1
          else np.zeros_like(image))
    dx = (np.gradient(image, axis=1) if image.shape[1] > 1
          else np.zeros_like(image))
    return np.hypot(dx, dy)


def _local_contrast(image: np.ndarray, sigma: float) -> np.ndarray:
    mean = gaussian_filter(image, sigma, mode="reflect")
    mean_square = gaussian_filter(image * image, sigma, mode="reflect")
    return np.sqrt(np.maximum(mean_square - mean * mean, 0.0))


def _oriented_spectrum(image: np.ndarray, orientations: int) -> np.ndarray:
    height, width = image.shape
    window = np.outer(np.hanning(height), np.hanning(width))
    transformed = np.fft.fft2((image - np.mean(image)) * window)
    power = np.abs(transformed) ** 2
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    angle = np.mod(np.arctan2(fy, fx), np.pi)
    radius = np.hypot(fy, fx)
    indices = np.minimum((angle * orientations / np.pi).astype(int), orientations - 1)
    valid = radius > 0
    return np.bincount(indices[valid].ravel(), weights=power[valid].ravel(),
                       minlength=orientations).astype(float)


def compare_measurements(target: np.ndarray, result: np.ndarray,
                         orientations: int = 8,
                         contrast_scales=DEFAULT_CONTRAST_SCALES) -> dict:
    """Compare absolute contrast, gradient, and directional spectral energy."""
    target, result = _images(target, result)
    if isinstance(orientations, bool) or not isinstance(orientations, int) or orientations < 2:
        raise ValueError("orientations must be an integer of at least two")
    scales = tuple(float(value) for value in contrast_scales)
    if not scales or not all(np.isfinite(scales)) or any(value <= 0 for value in scales):
        raise ValueError("contrast scales must be finite and positive")

    target_gradient = _gradient_magnitude(target)
    result_gradient = _gradient_magnitude(result)
    edge_threshold = float(np.percentile(target_gradient, 90))
    has_target_edges = edge_threshold > np.finfo(float).eps
    summaries = []

    def add(name: str, target_value: float, result_value: float, unit: str = "") -> None:
        summaries.append({"name": name, "target": float(target_value),
                          "result": float(result_value),
                          "ratio": _ratio(float(target_value), float(result_value)),
                          "unit": unit})

    add("Mean", np.mean(target), np.mean(result))
    add("Standard deviation", np.std(target), np.std(result))
    add("Gradient RMS", np.sqrt(np.mean(target_gradient ** 2)),
        np.sqrt(np.mean(result_gradient ** 2)))
    add("Gradient p95", np.percentile(target_gradient, 95),
        np.percentile(result_gradient, 95))
    add("Strong-edge density",
        np.mean(target_gradient >= edge_threshold) if has_target_edges else 0.0,
        np.mean(result_gradient >= edge_threshold) if has_target_edges else 0.0,
        "fraction")

    contrast = []
    for sigma in scales:
        first = _local_contrast(target, sigma)
        second = _local_contrast(result, sigma)
        target_mean, result_mean = float(np.mean(first)), float(np.mean(second))
        target_p95, result_p95 = float(np.percentile(first, 95)), float(np.percentile(second, 95))
        contrast.append({"sigma": sigma, "target_mean": target_mean,
                         "result_mean": result_mean,
                         "mean_ratio": _ratio(target_mean, result_mean),
                         "target_p95": target_p95, "result_p95": result_p95,
                         "p95_ratio": _ratio(target_p95, result_p95)})

    target_oriented = _oriented_spectrum(target, orientations)
    result_oriented = _oriented_spectrum(result, orientations)
    target_total = max(float(np.sum(target_oriented)), np.finfo(float).tiny)
    result_total = max(float(np.sum(result_oriented)), np.finfo(float).tiny)
    wedge_width = 180.0 / orientations
    oriented = [{"angle_min": index * wedge_width,
                 "angle_max": (index + 1) * wedge_width,
                 "target_energy": float(target_oriented[index]),
                 "result_energy": float(result_oriented[index]),
                 "ratio": _ratio(target_oriented[index], result_oriented[index]),
                 "target_fraction": float(target_oriented[index] / target_total),
                 "result_fraction": float(result_oriented[index] / result_total)}
                for index in range(orientations)]
    return {"shape": list(target.shape), "edge_threshold": edge_threshold,
            "summary": summaries, "local_contrast": contrast,
            "oriented_spectrum": oriented}
