"""Translation-tolerant statistics for procedural texture matching."""
from __future__ import annotations
from dataclasses import dataclass
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

    def __post_init__(self):
        values = (self.spectrum, self.histogram, self.autocorrelation, self.gradient, self.mse)
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

def _histogram_feature(image: np.ndarray, bins: int = 64) -> np.ndarray:
    histogram, _ = np.histogram(np.clip(image, 0, 1), bins=bins, range=(0, 1), density=False)
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

class TextureLoss:
    """Caches reference features and evaluates a weighted texture loss."""
    def __init__(self, reference: np.ndarray, weights: TextureLossWeights | None = None):
        self.reference = np.asarray(reference, dtype=np.float64)
        if self.reference.ndim != 2 or self.reference.size == 0:
            raise ValueError("texture loss expects a non-empty 2D image")
        self.weights = weights or TextureLossWeights()
        self._spectrum = _spectrum_features(self.reference)
        self._histogram = _histogram_feature(self.reference)
        self._autocorrelation = _autocorrelation_feature(self.reference)
        self._gradient = _gradient_feature(self.reference)

    def components(self, candidate: np.ndarray) -> dict[str, float]:
        image = np.asarray(candidate, dtype=np.float64)
        if image.shape != self.reference.shape or not np.all(np.isfinite(image)):
            raise ValueError("candidate must be finite and match the reference shape")
        spectra = _spectrum_features(image)
        spectrum = float(np.mean([np.mean((a-b) ** 2) for a, b in zip(self._spectrum, spectra)]))
        histogram = float(np.mean(np.abs(self._histogram - _histogram_feature(image))))
        autocorrelation = float(np.mean((self._autocorrelation - _autocorrelation_feature(image)) ** 2))
        gradient = float(np.mean(np.abs(self._gradient - _gradient_feature(image))))
        mse = float(np.mean((self.reference - image) ** 2))
        return {"spectrum_loss": spectrum, "histogram_loss": histogram,
                "autocorrelation_loss": autocorrelation, "gradient_loss": gradient,
                "mse_loss": mse}

    def evaluate(self, candidate: np.ndarray) -> tuple[float, dict[str, float]]:
        parts = self.components(candidate)
        weighted = (self.weights.spectrum * parts["spectrum_loss"]
                    + self.weights.histogram * parts["histogram_loss"]
                    + self.weights.autocorrelation * parts["autocorrelation_loss"]
                    + self.weights.gradient * parts["gradient_loss"]
                    + self.weights.mse * parts["mse_loss"])
        total = float(weighted / (self.weights.spectrum + self.weights.histogram
                                  + self.weights.autocorrelation + self.weights.gradient
                                  + self.weights.mse))
        return total, {"texture_loss": total, **parts}

def calculate_texture_loss(reference, candidate, weights: TextureLossWeights | None = None) -> dict[str, float]:
    """Return total and component texture losses for two scalar fields."""
    _, result = TextureLoss(np.asarray(reference), weights).evaluate(np.asarray(candidate))
    return result
