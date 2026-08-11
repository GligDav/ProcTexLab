"""Stable, client-facing fitting API."""
from __future__ import annotations
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Callable
import math
import numpy as np
from .fitting import fit_texture
from .io import normalize_image
from .metrics import calculate_metrics
from .model import ProceduralTextureModel
from .texture_loss import TextureLossWeights, calculate_texture_loss

ProgressCallback = Callable[[str, float, str], None]
CancelCallback = Callable[[], bool]

@dataclass(frozen=True)
class FitConfig:
    """Bounded controls for sparse multiscale fitting."""
    seed: int = 0
    max_components: int = 8
    max_iterations: int = 40
    fitting_resolution: int | None = 96
    component_families: tuple[str, ...] = ("sinusoid", "gabor", "gaussian_rbf")
    fft_candidates: int = 24
    min_frequency: float = 0.5
    max_frequency: float = 24.0
    min_improvement: float = 1e-6
    ridge: float = 1e-8
    fit_plane: bool = True
    spectrum_weight: float = 1.0
    histogram_weight: float = 0.5
    autocorrelation_weight: float = 0.75
    gradient_weight: float = 0.5
    def __post_init__(self):
        allowed = {"sinusoid", "gabor", "gaussian_rbf"}
        if self.max_components < 0 or self.max_iterations < 1 or self.fft_candidates < 1:
            raise ValueError("component/iteration/candidate counts are invalid")
        if self.fitting_resolution is not None and self.fitting_resolution < 8:
            raise ValueError("fitting_resolution must be at least 8 or None")
        if self.min_frequency < 0 or self.max_frequency <= self.min_frequency:
            raise ValueError("frequency bounds are invalid")
        if not math.isfinite(self.min_improvement) or self.min_improvement < 0:
            raise ValueError("min_improvement must be a finite, non-negative number")
        if not set(self.component_families) <= allowed:
            raise ValueError("unsupported component family")
        TextureLossWeights(self.spectrum_weight, self.histogram_weight,
                           self.autocorrelation_weight, self.gradient_weight)

    @property
    def texture_loss_weights(self) -> TextureLossWeights:
        """Return the validated composite texture-loss weights."""
        return TextureLossWeights(self.spectrum_weight, self.histogram_weight,
                                  self.autocorrelation_weight, self.gradient_weight)

@dataclass
class FitResult:
    """Fitted model plus full-resolution diagnostics."""
    model: ProceduralTextureModel
    metrics: dict[str, float]
    reconstruction: np.ndarray = field(repr=False)
    residual: np.ndarray = field(repr=False)
    metadata: dict = field(default_factory=dict)
    def evaluate(self, width: int, height: int) -> np.ndarray:
        return self.model.evaluate(width, height)
    def evaluate_region(
        self, width: int, height: int,
        u_bounds: tuple[float, float] = (0.0, 1.0),
        v_bounds: tuple[float, float] = (0.0, 1.0),
    ) -> np.ndarray:
        """Evaluate the fitted model outside its original UV domain."""
        return self.model.evaluate_region(width, height, u_bounds, v_bounds)
    def to_dict(self) -> dict:
        return {"schema_version": 1, "model": self.model.to_dict(), "metrics": self.metrics,
                "metadata": self.metadata}
    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

class TextureFitter:
    """Fit scalar rasters to a sparse procedural model."""
    def __init__(self, config: FitConfig | None = None): self.config = config or FitConfig()
    def fit(self, image_array: np.ndarray, progress_callback: ProgressCallback | None = None,
            cancel_callback: CancelCallback | None = None) -> FitResult:
        target = normalize_image(np.asarray(image_array))
        model, metadata = fit_texture(target, self.config, progress_callback, cancel_callback)
        reconstruction = model.evaluate(target.shape[1], target.shape[0])
        metrics = calculate_metrics(target, reconstruction)
        metrics.update(calculate_texture_loss(target, reconstruction, self.config.texture_loss_weights))
        return FitResult(model, metrics, reconstruction,
                         target - reconstruction, metadata)
