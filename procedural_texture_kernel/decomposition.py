"""Reconstructable image decompositions used by the fitting objective."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter


class ImageDecomposition(ABC):
    """Interface for same-resolution, additive image decompositions."""

    @abstractmethod
    def decompose(self, image: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return bands which add up to ``image``."""

    def reconstruct(self, bands: tuple[np.ndarray, ...] | list[np.ndarray]) -> np.ndarray:
        if not bands:
            raise ValueError("at least one band is required")
        shape = np.asarray(bands[0]).shape
        if any(np.asarray(band).shape != shape for band in bands):
            raise ValueError("all bands must have the same shape")
        return np.sum(np.stack(bands), axis=0, dtype=np.float64)


@dataclass(frozen=True)
class LaplacianPyramid(ImageDecomposition):
    """Full-resolution Laplacian bands separated by octave-spaced Gaussians.

    The first bands contain progressively lower-frequency differences and the
    final band is the Gaussian residual. Keeping all bands at the input size
    avoids resampling artifacts and is convenient for Blender-oriented fitting.
    """

    bands: int = 5
    base_sigma: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.bands, bool) or not isinstance(self.bands, int) or self.bands < 1:
            raise ValueError("bands must be a positive integer")
        if not np.isfinite(self.base_sigma) or self.base_sigma <= 0:
            raise ValueError("base_sigma must be finite and positive")

    @property
    def sigmas(self) -> tuple[float, ...]:
        return tuple(self.base_sigma * (2.0 ** index) for index in range(self.bands - 1))

    def decompose(self, image: np.ndarray) -> tuple[np.ndarray, ...]:
        source = np.asarray(image, dtype=np.float64)
        if source.ndim != 2 or source.size == 0 or not np.all(np.isfinite(source)):
            raise ValueError("decomposition expects a finite, non-empty 2D image")
        if self.bands == 1:
            return (source.copy(),)
        blurred = tuple(gaussian_filter(source, sigma, mode="reflect") for sigma in self.sigmas)
        return (source - blurred[0],) + tuple(
            blurred[index] - blurred[index + 1] for index in range(len(blurred) - 1)
        ) + (blurred[-1],)


def create_decomposition(method: str, bands: int = 5,
                         base_sigma: float = 1.0) -> ImageDecomposition:
    """Construct a decomposition by name, leaving room for future methods."""
    if method == "laplacian":
        return LaplacianPyramid(bands=bands, base_sigma=base_sigma)
    raise ValueError(f"unsupported decomposition method: {method!r}")
