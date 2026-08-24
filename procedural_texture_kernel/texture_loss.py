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
    local_contrast: float = 0.0
    absolute_spectrum: float = 0.0
    oriented_spectrum: float = 0.0

    def __post_init__(self) -> None:
        """Reject negative, non-finite, or collectively zero loss weights."""
        values = (self.spectrum, self.histogram, self.autocorrelation, self.gradient,
                  self.mse, self.local_structure, self.local_contrast,
                  self.absolute_spectrum, self.oriented_spectrum)
        if not all(np.isfinite(values)) or any(value < 0 for value in values):
            raise ValueError("texture loss weights must be finite and non-negative")
        if sum(values) <= 0:
            raise ValueError("at least one texture loss weight must be positive")

def _spectrum_features(image: np.ndarray) -> list[np.ndarray]:
    """Return log-power Fourier descriptors across a small image pyramid."""
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


_ABSOLUTE_SPECTRUM_EDGES = np.array((0.0, .125, .25, .5, .75,
                                     np.sqrt(2.0) + 1e-12))


def _absolute_spectrum_energy(image: np.ndarray) -> np.ndarray:
    """Return window-corrected absolute energy in Nyquist-relative bands."""
    height, width = image.shape
    window = np.outer(np.hanning(height), np.hanning(width))
    window_power = max(float(np.mean(window * window)), np.finfo(float).tiny)
    transformed = np.fft.fft2((image - np.mean(image)) * window)
    power = np.abs(transformed) ** 2 / (image.size ** 2 * window_power)
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    radius = np.hypot(fy, fx) / .5
    return np.asarray([np.sum(power[(radius >= lower) & (radius < upper)])
                       for lower, upper in zip(_ABSOLUTE_SPECTRUM_EDGES[:-1],
                                               _ABSOLUTE_SPECTRUM_EDGES[1:])],
                      dtype=np.float64)


def _oriented_spectrum_energy(image: np.ndarray, orientations: int = 8) -> np.ndarray:
    """Return absolute energy for radial bands split into orientation wedges."""
    height, width = image.shape
    window = np.outer(np.hanning(height), np.hanning(width))
    window_power = max(float(np.mean(window * window)), np.finfo(float).tiny)
    transformed = np.fft.fft2((image - np.mean(image)) * window)
    power = np.abs(transformed) ** 2 / (image.size ** 2 * window_power)
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    radius = np.hypot(fy, fx) / .5
    angle = np.mod(np.arctan2(fy, fx), np.pi)
    wedge = np.minimum((angle * orientations / np.pi).astype(int), orientations - 1)
    result = np.zeros((len(_ABSOLUTE_SPECTRUM_EDGES) - 1, orientations))
    for band, (lower, upper) in enumerate(zip(_ABSOLUTE_SPECTRUM_EDGES[:-1],
                                               _ABSOLUTE_SPECTRUM_EDGES[1:])):
        radial_mask = (radius >= lower) & (radius < upper)
        for orientation in range(orientations):
            result[band, orientation] = np.sum(
                power[radial_mask & (wedge == orientation)])
    return result

def _histogram_feature(image: np.ndarray, value_range: tuple[float, float],
                       bins: int = 64) -> np.ndarray:
    """Return a normalized cumulative histogram over ``value_range``."""
    histogram, _ = np.histogram(np.clip(image, *value_range), bins=bins,
                                range=value_range, density=False)
    return np.cumsum(histogram, dtype=np.float64) / image.size

def _autocorrelation_feature(image: np.ndarray) -> np.ndarray:
    """Return the central normalized cyclic autocorrelation neighborhood."""
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

def _gradient_feature(
    image: np.ndarray, normalizers: tuple[float, ...] | None = None
) -> tuple[np.ndarray, tuple[float, ...]]:
    """Return multiscale edge statistics using shared reference normalization."""
    features, used_normalizers = [], []
    for index, sigma in enumerate((0.0, 1.0, 2.0, 4.0)):
        working = image if sigma == 0 else gaussian_filter(image, sigma)
        # Periodic differences retain cyclic-translation invariance.
        dx = np.roll(working, -1, axis=1) - working
        dy = np.roll(working, -1, axis=0) - working
        magnitude = np.hypot(dx, dy)
        scale = (max(float(np.percentile(magnitude, 99)), 1e-12)
                 if normalizers is None else normalizers[index])
        used_normalizers.append(scale)
        magnitude_hist, _ = np.histogram(
            np.clip(magnitude / scale, 0, 1), bins=32, range=(0, 1))
        angles = np.mod(np.arctan2(dy, dx), np.pi)
        orientation_hist, _ = np.histogram(
            angles, bins=16, range=(0, np.pi), weights=magnitude)
        orientation_hist = orientation_hist.astype(float)
        orientation_hist /= max(float(orientation_hist.sum()), 1e-12)
        features.extend((float(np.mean(magnitude)), float(np.std(magnitude))))
        features.extend((magnitude_hist.astype(float) / image.size).tolist())
        features.extend(orientation_hist.tolist())
    return np.asarray(features), tuple(used_normalizers)


def _local_contrast_feature(
    image: np.ndarray, normalizers: tuple[float, ...] | None = None
) -> tuple[np.ndarray, tuple[float, ...]]:
    """Describe local standard-deviation distributions at several scales."""
    features, used_normalizers = [], []
    squared = np.asarray(image, dtype=np.float64) ** 2
    for index, sigma in enumerate((1.0, 2.0, 4.0, 8.0)):
        mean = gaussian_filter(image, sigma)
        variance = np.maximum(gaussian_filter(squared, sigma) - mean * mean, 0.0)
        contrast = np.sqrt(variance)
        scale = (max(float(np.percentile(contrast, 99)), 1e-12)
                 if normalizers is None else normalizers[index])
        used_normalizers.append(scale)
        histogram, _ = np.histogram(
            np.clip(contrast / scale, 0, 1), bins=32, range=(0, 1))
        features.extend((float(np.mean(contrast)), float(np.std(contrast))))
        features.extend((histogram.astype(float) / image.size).tolist())
    return np.asarray(features), tuple(used_normalizers)


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
    """Return normalized correlation between flattened centered arrays."""
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
                 local_structure_block_size: int = 8) -> None:
        """Precompute reference features for repeated candidate evaluation."""
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
        self._absolute_spectrum = _absolute_spectrum_energy(self.reference)
        self._absolute_spectrum_epsilon = max(
            float(np.sum(self._absolute_spectrum)) * 1e-8, 1e-16)
        self._oriented_spectrum = _oriented_spectrum_energy(self.reference)
        self._oriented_spectrum_epsilon = max(
            float(np.sum(self._oriented_spectrum)) * 1e-8
            / self._oriented_spectrum.size, 1e-16)
        self._histogram = _histogram_feature(self.reference, self._histogram_range)
        self._autocorrelation = _autocorrelation_feature(self.reference)
        self._gradient, self._gradient_normalizers = _gradient_feature(self.reference)
        if self.weights.local_contrast > 0:
            self._local_contrast, self._contrast_normalizers = _local_contrast_feature(
                self.reference)
        else:
            self._local_contrast, self._contrast_normalizers = None, None
        self._local_structure_settings = (local_structure_scales,
                                          local_structure_orientations,
                                          local_structure_block_size)
        self._local_structure = (_local_structure_feature(
            self.reference, *self._local_structure_settings)
            if self.weights.local_structure > 0 else None)

    def components(self, candidate: np.ndarray) -> dict[str, float]:
        """Return every unweighted diagnostic loss for a matching candidate."""
        image = np.asarray(candidate, dtype=np.float64)
        if image.shape != self.reference.shape or not np.all(np.isfinite(image)):
            raise ValueError("candidate must be finite and match the reference shape")
        # Compute all diagnostics here for reporting, irrespective of zero weights.
        spectra = _spectrum_features(image)
        spectrum = float(np.mean([np.mean((a-b) ** 2) for a, b in zip(self._spectrum, spectra)]))
        candidate_energy = _absolute_spectrum_energy(image)
        log_ratio = np.log10((candidate_energy + self._absolute_spectrum_epsilon)
                             / (self._absolute_spectrum
                                + self._absolute_spectrum_epsilon)) / 8.0
        absolute_spectrum = float(np.mean(log_ratio * log_ratio))
        candidate_oriented = _oriented_spectrum_energy(image)
        oriented_log_ratio = np.log10(
            (candidate_oriented + self._oriented_spectrum_epsilon)
            / (self._oriented_spectrum + self._oriented_spectrum_epsilon)) / 8.0
        oriented_spectrum = float(np.mean(oriented_log_ratio * oriented_log_ratio))
        histogram = float(np.mean(np.abs(
            self._histogram - _histogram_feature(image, self._histogram_range))))
        autocorrelation = float(np.mean((self._autocorrelation - _autocorrelation_feature(image)) ** 2))
        candidate_gradient, _ = _gradient_feature(image, self._gradient_normalizers)
        gradient = float(np.mean(np.abs(self._gradient - candidate_gradient)))
        local_contrast = 0.0
        if self._local_contrast is not None:
            candidate_contrast, _ = _local_contrast_feature(
                image, self._contrast_normalizers)
            local_contrast = float(np.mean(np.abs(
                self._local_contrast - candidate_contrast)))
        local_structure = 0.0
        if self._local_structure is not None:
            candidate_structure = _local_structure_feature(
                image, *self._local_structure_settings)
            local_structure = float(np.mean(
                (self._local_structure - candidate_structure) ** 2))
        mse = float(np.mean((self.reference - image) ** 2))
        return {"spectrum_loss": spectrum,
                "absolute_spectrum_loss": absolute_spectrum,
                "oriented_spectrum_loss": oriented_spectrum,
                "histogram_loss": histogram,
                "autocorrelation_loss": autocorrelation, "gradient_loss": gradient,
                "local_structure_loss": local_structure,
                "local_contrast_loss": local_contrast,
                "mse_loss": mse}

    def evaluate(self, candidate: np.ndarray) -> tuple[float, dict[str, float]]:
        """Return the weighted total and all named component losses."""
        parts = self.components(candidate)
        total = self._weighted_total(parts)
        return total, {"texture_loss": total, **parts}

    def _weighted_total(self, parts: dict[str, float]) -> float:
        """Combine available component losses using normalized configured weights."""
        # Normalize by total weight so only relative weight magnitudes matter.
        weighted = (self.weights.spectrum * parts.get("spectrum_loss", 0.0)
                    + self.weights.absolute_spectrum
                    * parts.get("absolute_spectrum_loss", 0.0)
                    + self.weights.oriented_spectrum
                    * parts.get("oriented_spectrum_loss", 0.0)
                    + self.weights.histogram * parts.get("histogram_loss", 0.0)
                    + self.weights.autocorrelation
                    * parts.get("autocorrelation_loss", 0.0)
                    + self.weights.gradient * parts.get("gradient_loss", 0.0)
                    + self.weights.mse * parts.get("mse_loss", 0.0)
                    + self.weights.local_structure
                    * parts.get("local_structure_loss", 0.0)
                    + self.weights.local_contrast
                    * parts.get("local_contrast_loss", 0.0))
        total_denominator = (self.weights.spectrum + self.weights.histogram
                             + self.weights.autocorrelation + self.weights.gradient
                             + self.weights.mse + self.weights.local_structure
                             + self.weights.local_contrast
                             + self.weights.absolute_spectrum)
        total_denominator += self.weights.oriented_spectrum
        return float(weighted / total_denominator)

    def evaluate_total(self, candidate: np.ndarray) -> float:
        """Evaluate only enabled terms for optimizer and candidate inner loops.

        ``evaluate`` intentionally continues to calculate every diagnostic loss
        for the public API.  Fitting only needs the weighted scalar for most
        probes, so skipping zero-weight features avoids unnecessary FFTs and
        Gaussian filters without changing the objective.
        """
        image = np.asarray(candidate, dtype=np.float64)
        if image.shape != self.reference.shape or not np.all(np.isfinite(image)):
            raise ValueError("candidate must be finite and match the reference shape")
        parts: dict[str, float] = {}
        if self.weights.spectrum:
            spectra = _spectrum_features(image)
            parts["spectrum_loss"] = float(np.mean([
                np.mean((a - b) ** 2) for a, b in zip(self._spectrum, spectra)]))
        if self.weights.absolute_spectrum:
            energy = _absolute_spectrum_energy(image)
            ratio = np.log10((energy + self._absolute_spectrum_epsilon)
                             / (self._absolute_spectrum
                                + self._absolute_spectrum_epsilon)) / 8.0
            parts["absolute_spectrum_loss"] = float(np.mean(ratio * ratio))
        if self.weights.oriented_spectrum:
            energy = _oriented_spectrum_energy(image)
            ratio = np.log10((energy + self._oriented_spectrum_epsilon)
                             / (self._oriented_spectrum
                                + self._oriented_spectrum_epsilon)) / 8.0
            parts["oriented_spectrum_loss"] = float(np.mean(ratio * ratio))
        if self.weights.histogram:
            parts["histogram_loss"] = float(np.mean(np.abs(
                self._histogram - _histogram_feature(image, self._histogram_range))))
        if self.weights.autocorrelation:
            parts["autocorrelation_loss"] = float(np.mean((
                self._autocorrelation - _autocorrelation_feature(image)) ** 2))
        if self.weights.gradient:
            feature, _ = _gradient_feature(image, self._gradient_normalizers)
            parts["gradient_loss"] = float(np.mean(np.abs(self._gradient - feature)))
        if self.weights.local_structure:
            feature = _local_structure_feature(image, *self._local_structure_settings)
            parts["local_structure_loss"] = float(np.mean(
                (self._local_structure - feature) ** 2))
        if self.weights.local_contrast:
            feature, _ = _local_contrast_feature(image, self._contrast_normalizers)
            parts["local_contrast_loss"] = float(np.mean(np.abs(
                self._local_contrast - feature)))
        if self.weights.mse:
            parts["mse_loss"] = float(np.mean((self.reference - image) ** 2))
        # Missing terms have zero weights and therefore cannot affect the sum.
        return self._weighted_total(parts)

def calculate_texture_loss(
    reference: np.ndarray, candidate: np.ndarray,
    weights: TextureLossWeights | None = None, **settings: object,
) -> dict[str, float]:
    """Return total and component texture losses for two scalar fields.

    ``reference`` and ``candidate`` must be equally shaped finite 2D arrays.
    Extra keyword settings configure :class:`TextureLoss`; the returned mapping
    contains ``texture_loss`` plus each individual diagnostic term.
    """
    _, result = TextureLoss(np.asarray(reference), weights, **settings).evaluate(np.asarray(candidate))
    return result
