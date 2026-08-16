"""Translation-tolerant statistics for procedural texture matching."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from scipy.ndimage import gaussian_filter

@dataclass(frozen=True)
class TextureLossWeights:
    """Weights of the composite statistical and pixel-aligned objective."""
    spectrum: float = 1.0
    histogram: float = 0.5
    autocorrelation: float = 0.75
    gradient: float = 0.5
    mse: float = 1.0
    local_structure: float = 0.0

    def __post_init__(self):
        values = (self.spectrum, self.histogram, self.autocorrelation, self.gradient,
                  self.mse, self.local_structure)
        if not all(np.isfinite(values)) or any(value < 0 for value in values):
            raise ValueError("texture loss weights must be finite and non-negative")
        if sum(values) <= 0:
            raise ValueError("at least one texture loss weight must be positive")

def _spectrum_features(image: np.ndarray) -> list[np.ndarray]:
    features = []
    current = np.asarray(image, dtype=np.float64)
    for _ in range(3):
        centered = current - np.mean(current)
        power = np.abs(np.fft.rfft2(centered)) ** 2
        power /= max(float(np.sum(power)), 1e-12)
        features.append(np.log1p(power * power.size))
        if min(current.shape) < 16:
            break
        current = gaussian_filter(current, 1.0)[::2, ::2]
    return features

def _histogram_feature(image: np.ndarray, value_range: tuple[float, float],
                       bins: int = 64) -> np.ndarray:
    histogram, _ = np.histogram(np.clip(image, *value_range), bins=bins,
                                range=value_range, density=False)
    return np.cumsum(histogram, dtype=np.float64) / image.size

def _autocorrelation_feature(image: np.ndarray) -> np.ndarray:
    centered = image - np.mean(image)
    variance = float(np.mean(centered * centered))
    radius_y = min(8, image.shape[0] // 2)
    radius_x = min(8, image.shape[1] // 2)
    if variance < 1e-12:
        return np.zeros((2 * radius_y + 1, 2 * radius_x + 1), dtype=np.float64)
    correlation = np.fft.ifft2(np.abs(np.fft.fft2(centered)) ** 2).real / (image.size * variance)
    correlation = np.fft.fftshift(correlation)
    cy, cx = image.shape[0] // 2, image.shape[1] // 2
    return correlation[cy-radius_y:cy+radius_y+1, cx-radius_x:cx+radius_x+1]

def _gradient_feature(image: np.ndarray) -> np.ndarray:
    # Periodic differences make these statistics invariant to cyclic translation.
    dx = np.roll(image, -1, axis=1) - image
    dy = np.roll(image, -1, axis=0) - image
    magnitude = np.hypot(dx, dy)
    scale = max(float(np.percentile(magnitude, 99)), 1e-12)
    magnitude_hist, _ = np.histogram(np.clip(magnitude / scale, 0, 1), bins=32, range=(0, 1))
    angles = np.mod(np.arctan2(dy, dx), np.pi)
    orientation_hist, _ = np.histogram(angles, bins=16, range=(0, np.pi), weights=magnitude)
    magnitude_hist = magnitude_hist.astype(float) / image.size
    orientation_hist = orientation_hist.astype(float)
    orientation_hist /= max(float(orientation_hist.sum()), 1e-12)
    return np.concatenate(([np.mean(magnitude), np.std(magnitude)], magnitude_hist, orientation_hist))


@lru_cache(maxsize=32)
def _oriented_filter_bank(shape: tuple[int, int], scales: int,
                          orientations: int) -> tuple[np.ndarray, ...]:
    """Construct one-sided Gaussian frequency filters with complex responses."""
    height, width = shape
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    filters = []
    for scale in range(scales):
        center = .25 / (2 ** scale)
        bandwidth = max(center * .55, 1.0 / max(height, width))
        for orientation in range(orientations):
            angle = np.pi * orientation / orientations
            center_x = center * np.cos(angle)
            center_y = center * np.sin(angle)
            distance = (fx - center_x) ** 2 + (fy - center_y) ** 2
            filters.append(np.exp(-distance / (2 * bandwidth * bandwidth)))
    return tuple(filters)


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    first = a.ravel() - np.mean(a)
    second = b.ravel() - np.mean(b)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return 0.0 if denominator < 1e-12 else float(np.vdot(first, second).real / denominator)


def _local_structure_feature(image: np.ndarray, scales: int,
                             orientations: int, block_size: int) -> np.ndarray:
    """Describe oriented magnitudes and adjacent-scale phase relationships."""
    centered = np.asarray(image, dtype=np.float64) - np.mean(image)
    standard_deviation = float(np.std(centered))
    if standard_deviation < 1e-12:
        # Keep a stable feature length by processing an all-zero normalized image.
        normalized = np.zeros_like(centered)
    else:
        normalized = centered / standard_deviation
    transformed = np.fft.fft2(normalized)
    responses = [np.fft.ifft2(transformed * frequency_filter)
                 for frequency_filter in _oriented_filter_bank(
                     normalized.shape, scales, orientations)]
    magnitudes = [np.abs(response) for response in responses]
    features: list[float] = []
    quantiles = np.linspace(0, 1, 9)
    for magnitude in magnitudes:
        features.extend((float(np.mean(magnitude)), float(np.std(magnitude))))
        block_means = []
        for top in range(0, magnitude.shape[0], block_size):
            for left in range(0, magnitude.shape[1], block_size):
                block = magnitude[top:top + block_size, left:left + block_size]
                block_means.append(float(np.mean(block)))
        features.extend(np.quantile(block_means, quantiles).tolist())
    for scale in range(scales):
        offset = scale * orientations
        for orientation in range(orientations):
            adjacent = (orientation + 1) % orientations
            features.append(_correlation(magnitudes[offset + orientation],
                                         magnitudes[offset + adjacent]))
    epsilon = 1e-12
    for scale in range(scales - 1):
        child_offset = scale * orientations
        parent_offset = (scale + 1) * orientations
        for orientation in range(orientations):
            child_index = child_offset + orientation
            parent_index = parent_offset + orientation
            child = responses[child_index]
            parent = responses[parent_index]
            features.append(_correlation(magnitudes[child_index], magnitudes[parent_index]))
            phase_relation = child * np.conj(parent)
            weights = np.sqrt(magnitudes[child_index] * magnitudes[parent_index])
            coherence = np.sum(weights * phase_relation / (np.abs(phase_relation) + epsilon))
            coherence /= max(float(np.sum(weights)), epsilon)
            features.extend((float(coherence.real), float(coherence.imag)))
    return np.asarray(features, dtype=np.float64)

class TextureLoss:
    """Caches reference features and evaluates a weighted texture loss."""
    def __init__(self, reference: np.ndarray, weights: TextureLossWeights | None = None,
                 local_structure_scales: int = 3, local_structure_orientations: int = 4,
                 local_structure_block_size: int = 8):
        self.reference = np.asarray(reference, dtype=np.float64)
        if (self.reference.ndim != 2 or self.reference.size == 0
                or not np.all(np.isfinite(self.reference))):
            raise ValueError("texture loss expects a non-empty 2D image")
        for value, name in ((local_structure_scales, "scales"),
                            (local_structure_orientations, "orientations"),
                            (local_structure_block_size, "block size")):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"local structure {name} must be a positive integer")
        self.weights = weights or TextureLossWeights()
        low, high = float(np.min(self.reference)), float(np.max(self.reference))
        self._histogram_range = (low, high) if high > low else (low - .5, high + .5)
        self._spectrum = _spectrum_features(self.reference)
        self._histogram = _histogram_feature(self.reference, self._histogram_range)
        self._autocorrelation = _autocorrelation_feature(self.reference)
        self._gradient = _gradient_feature(self.reference)
        self._local_structure_settings = (local_structure_scales,
                                          local_structure_orientations,
                                          local_structure_block_size)
        self._local_structure = (_local_structure_feature(
            self.reference, *self._local_structure_settings)
            if self.weights.local_structure > 0 else None)

    def components(self, candidate: np.ndarray) -> dict[str, float]:
        image = np.asarray(candidate, dtype=np.float64)
        if image.shape != self.reference.shape or not np.all(np.isfinite(image)):
            raise ValueError("candidate must be finite and match the reference shape")
        spectra = _spectrum_features(image)
        spectrum = float(np.mean([np.mean((a-b) ** 2) for a, b in zip(self._spectrum, spectra)]))
        histogram = float(np.mean(np.abs(
            self._histogram - _histogram_feature(image, self._histogram_range))))
        autocorrelation = float(np.mean((self._autocorrelation - _autocorrelation_feature(image)) ** 2))
        gradient = float(np.mean(np.abs(self._gradient - _gradient_feature(image))))
        local_structure = 0.0
        if self._local_structure is not None:
            candidate_structure = _local_structure_feature(
                image, *self._local_structure_settings)
            local_structure = float(np.mean(
                (self._local_structure - candidate_structure) ** 2))
        mse = float(np.mean((self.reference - image) ** 2))
        return {"spectrum_loss": spectrum, "histogram_loss": histogram,
                "autocorrelation_loss": autocorrelation, "gradient_loss": gradient,
                "local_structure_loss": local_structure,
                "mse_loss": mse}

    def evaluate(self, candidate: np.ndarray) -> tuple[float, dict[str, float]]:
        parts = self.components(candidate)
        weighted = (self.weights.spectrum * parts["spectrum_loss"]
                    + self.weights.histogram * parts["histogram_loss"]
                    + self.weights.autocorrelation * parts["autocorrelation_loss"]
                    + self.weights.gradient * parts["gradient_loss"]
                    + self.weights.mse * parts["mse_loss"]
                    + self.weights.local_structure * parts["local_structure_loss"])
        total = float(weighted / (self.weights.spectrum + self.weights.histogram
                                  + self.weights.autocorrelation + self.weights.gradient
                                  + self.weights.mse + self.weights.local_structure))
        return total, {"texture_loss": total, **parts}

def calculate_texture_loss(reference, candidate, weights: TextureLossWeights | None = None,
                           **settings) -> dict[str, float]:
    """Return total and component texture losses for two scalar fields."""
    _, result = TextureLoss(np.asarray(reference), weights, **settings).evaluate(np.asarray(candidate))
    return result
