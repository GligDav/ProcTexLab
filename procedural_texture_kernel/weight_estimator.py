"""Heuristic loss-weight estimation for individual pyramid bands.

Feature extraction is deliberately independent of weight mapping.  The defaults
are deterministic starting points for experimentation, not learned or universally
optimal constants.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.ndimage import sobel


@dataclass(frozen=True)
class BandFeatures:
    """Normalized band descriptors (nominally 0..1) plus raw kurtosis."""

    spectral_entropy: float
    spectral_anisotropy: float
    autocorrelation_strength: float
    gradient_coherence: float
    kurtosis: float
    raw_excess_kurtosis: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Return all extracted descriptor values as a plain mapping."""
        return asdict(self)


@dataclass(frozen=True)
class LossWeights:
    """Normalized weights corresponding to the four statistical loss terms."""

    spectrum: float
    histogram: float
    autocorrelation: float
    gradient: float

    def to_dict(self) -> dict[str, float]:
        """Return the four normalized statistical loss weights by name."""
        return asdict(self)


@dataclass(frozen=True)
class WeightEstimatorResult:
    """Feature descriptors and the loss weights derived from them."""
    features: BandFeatures
    weights: LossWeights

    def to_dict(self) -> dict[str, dict[str, float]]:
        """Return nested serializable feature and weight mappings."""
        return {"features": self.features.to_dict(), "weights": self.weights.to_dict()}


@dataclass(frozen=True)
class WeightMappingConfig:
    """Coefficients for smooth feature-to-score mappings.

    Each tuple is ``(baseline, entropy, anisotropy, autocorrelation,
    coherence, kurtosis)``.  Scores are normalized to sum to one.
    """

    spectrum: tuple[float, ...] = (0.35, 0.65, 0.35, 0.00, 0.00, 0.00)
    histogram: tuple[float, ...] = (0.30, 0.25, 0.00, 0.00, 0.00, 0.65)
    autocorrelation: tuple[float, ...] = (0.25, 0.00, 0.00, 1.00, 0.00, 0.00)
    gradient: tuple[float, ...] = (0.25, 0.00, 0.30, 0.00, 0.80, 0.00)

    def __post_init__(self) -> None:
        """Validate every feature-to-weight coefficient vector."""
        for name in ("spectrum", "histogram", "autocorrelation", "gradient"):
            values = getattr(self, name)
            if len(values) != 6 or not np.all(np.isfinite(values)) or np.any(np.asarray(values) < 0):
                raise ValueError(f"{name} coefficients must contain six finite non-negative values")


@dataclass(frozen=True)
class WeightEstimatorConfig:
    """Numerical and heuristic settings for :class:`WeightEstimator`."""

    epsilon: float = 1e-12
    autocorrelation_percentile: float = 95.0
    kurtosis_scale: float = 6.0
    mapping: WeightMappingConfig = field(default_factory=WeightMappingConfig)

    def __post_init__(self) -> None:
        """Validate numerical tolerances and heuristic scale parameters."""
        if not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if not 0 <= self.autocorrelation_percentile <= 100:
            raise ValueError("autocorrelation_percentile must be in [0, 100]")
        if not np.isfinite(self.kurtosis_scale) or self.kurtosis_scale <= 0:
            raise ValueError("kurtosis_scale must be finite and positive")


class BandFeatureExtractor:
    """Extract descriptors while sharing one FFT across spectral/ACF metrics."""

    def __init__(self, config: WeightEstimatorConfig | None = None) -> None:
        """Create an extractor using ``config`` or deterministic defaults."""
        self.config = config or WeightEstimatorConfig()

    def extract(self, band: np.ndarray) -> BandFeatures:
        """Measure normalized spectral, correlation, edge, and kurtosis features."""
        image = np.asarray(band, dtype=np.float64)
        if image.ndim != 2 or image.size == 0:
            raise ValueError("band must be a non-empty 2D array")
        if not np.all(np.isfinite(image)):
            raise ValueError("band must contain only finite values")

        # Analyze a scale-normalized copy.  All descriptors are amplitude-scale
        # invariant, and this prevents overflow for unusually large float ranges.
        amplitude_scale = max(float(np.max(np.abs(image))), 1.0)
        working = image / amplitude_scale
        centered = working - float(np.mean(working))
        variance = float(np.mean(centered * centered))
        degenerate = variance <= self.config.epsilon
        if degenerate:
            return BandFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        fft = np.fft.fft2(centered)
        power = np.abs(fft) ** 2
        total_power = float(np.sum(power))
        entropy = self._spectral_entropy(power, total_power)
        anisotropy = self._spectral_anisotropy(power, total_power)
        autocorrelation = self._autocorrelation_strength(power, variance)
        coherence = self._gradient_coherence(working)
        raw_kurtosis = float(np.mean(centered ** 4) / (variance * variance) - 3.0)
        if not np.isfinite(raw_kurtosis):
            raw_kurtosis = 0.0
        kurtosis = float(np.tanh(abs(raw_kurtosis) / self.config.kurtosis_scale))
        return BandFeatures(entropy, anisotropy, autocorrelation, coherence,
                            kurtosis, raw_kurtosis)

    def _spectral_entropy(self, power: np.ndarray, total: float) -> float:
        """Return normalized Shannon entropy of a Fourier power array."""
        if total <= self.config.epsilon or power.size <= 1:
            return 0.0
        probabilities = power.ravel() / total
        positive = probabilities > self.config.epsilon / probabilities.size
        entropy = -float(np.sum(probabilities[positive] * np.log(probabilities[positive])))
        return float(np.clip(entropy / np.log(probabilities.size), 0.0, 1.0))

    def _spectral_anisotropy(self, power: np.ndarray, total: float) -> float:
        """Return directional imbalance derived from spectral second moments."""
        fy = np.fft.fftfreq(power.shape[0])
        fx = np.fft.fftfreq(power.shape[1])
        yy, xx = np.meshgrid(fy, fx, indexing="ij")
        mxx = float(np.sum(power * xx * xx) / total)
        myy = float(np.sum(power * yy * yy) / total)
        mxy = float(np.sum(power * xx * yy) / total)
        eigenvalues = np.linalg.eigvalsh(((mxx, mxy), (mxy, myy)))
        value = (eigenvalues[1] - eigenvalues[0]) / (eigenvalues.sum() + self.config.epsilon)
        return float(np.clip(value, 0.0, 1.0))

    def _autocorrelation_strength(self, power: np.ndarray, variance: float) -> float:
        """Summarize strong off-center normalized autocorrelation responses."""
        correlation = np.fft.fftshift(np.fft.ifft2(power).real / (power.size * variance))
        yy, xx = np.indices(correlation.shape)
        cy, cx = np.array(correlation.shape) // 2
        # Remove the center and its immediate neighbors: they mostly measure
        # smoothness, while a percentile resists isolated numerical spikes.
        off_center = correlation[(yy - cy) ** 2 + (xx - cx) ** 2 > 2]
        if off_center.size == 0:
            return 0.0
        strength = np.percentile(np.abs(off_center), self.config.autocorrelation_percentile)
        return float(np.clip(strength, 0.0, 1.0))

    def _gradient_coherence(self, image: np.ndarray) -> float:
        """Return the magnitude-weighted agreement of local edge directions."""
        if min(image.shape) < 2:
            return 0.0
        gx = sobel(image, axis=1, mode="reflect")
        gy = sobel(image, axis=0, mode="reflect")
        magnitude = np.hypot(gx, gy)
        denominator = float(np.sum(magnitude))
        if denominator <= self.config.epsilon:
            return 0.0
        # m*exp(2j theta) without explicitly forming theta.
        numerator = abs(np.sum((gx * gx - gy * gy + 2j * gx * gy)
                               / (magnitude + self.config.epsilon)))
        return float(np.clip(numerator / denominator, 0.0, 1.0))


class WeightEstimator:
    """Analyze one Gaussian/Laplacian band and produce normalized loss weights."""

    def __init__(self, config: WeightEstimatorConfig | None = None,
                 extractor: BandFeatureExtractor | None = None) -> None:
        """Create an estimator and optionally reuse a configured extractor."""
        self.config = config or WeightEstimatorConfig()
        self.extractor = extractor or BandFeatureExtractor(self.config)

    def estimate(self, features: BandFeatures) -> LossWeights:
        """Map band descriptors to non-negative weights that sum to one."""
        vector = np.array((1.0, features.spectral_entropy, features.spectral_anisotropy,
                           features.autocorrelation_strength, features.gradient_coherence,
                           features.kurtosis))
        mapping = self.config.mapping
        raw = np.array([np.dot(getattr(mapping, name), vector) for name in
                        ("spectrum", "histogram", "autocorrelation", "gradient")])
        raw = np.maximum(raw, 0.0)
        total = float(np.sum(raw))
        normalized = raw / total if total > self.config.epsilon else np.full(4, 0.25)
        # Assign the last value from the remainder to make the invariant as tight
        # as floating point permits.
        normalized[-1] = 1.0 - float(np.sum(normalized[:-1]))
        return LossWeights(*map(float, normalized))

    def analyze(self, band: np.ndarray) -> WeightEstimatorResult:
        """Extract ``band`` features and return them with estimated weights."""
        features = self.extractor.extract(band)
        return WeightEstimatorResult(features, self.estimate(features))
