"""Stable, client-facing fitting API."""
from __future__ import annotations
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Callable
import math
import numpy as np
from .fitting import fit_texture
from .decomposition import create_decomposition
from .io import normalize_image
from .metrics import calculate_metrics
from .model import ProceduralTextureModel
from .texture_loss import TextureLossWeights, calculate_texture_loss
from .spectral_diagnostics import compare_spectra

ProgressCallback = Callable[[str, float, str], None]
CancelCallback = Callable[[], bool]
SUPPORTED_COMPONENT_FAMILIES = (
    "sinusoid", "spectral_noise", "gabor", "gaussian_rbf", "perlin_noise",
    "thresholded_noise",
    "masked_noise", "wavelet",
    "shader_graph",
    "voronoi_noise", "fbm", "ridged_multifractal", "turbulence_noise",
    "domain_warped_noise", "warped_ridged_multifractal",
    "warped_ridge_detail",
    "anisotropic_gaussian", "line", "step_edge",
    "dog_log", "polynomial_trend", "radial_wave", "spiral_wave",
    "sparse_impulse", "binary_primitive", "simple_constant"
)
DEFAULT_DETAIL_COMPONENT_FAMILIES = (
    "sinusoid", "spectral_noise", "gabor", "wavelet", "dog_log",
    "sparse_impulse", "line"
)

@dataclass(frozen=True)
class FitConfig:
    """Bounded controls for sparse multiscale fitting."""
    seed: int = 0
    max_components: int = 12
    max_iterations: int = 60
    fitting_resolution: int | None = 192
    component_families: tuple[str, ...] = SUPPORTED_COMPONENT_FAMILIES
    fft_candidates: int = 24
    spectral_noise_modes: int = 32
    noise_seed_candidates: int = 4
    min_frequency: float = 0.5
    max_frequency: float | None = None
    min_improvement: float = 1e-6
    ridge: float = 1e-8
    fit_plane: bool = True
    adaptive_texture_weights: bool = True
    spectrum_weight: float = 1.0
    histogram_weight: float = 0.5
    autocorrelation_weight: float = 0.75
    gradient_weight: float = 0.5
    mse_weight: float = 1.0
    local_structure_weight: float = 0.0
    local_contrast_weight: float = 0.0
    absolute_spectrum_weight: float = 0.25
    oriented_spectrum_weight: float = 0.25
    local_structure_scales: int = 3
    local_structure_orientations: int = 4
    local_structure_block_size: int = 8
    local_structure_candidate_limit: int = 16
    decomposition_method: str = "laplacian"
    decomposition_bands: int = 5
    decomposition_base_sigma: float = 1.0
    band_workers: int = 1
    candidate_workers: int = 1
    compute_backend: str = "numpy"
    gpu_batch_size: int = 16
    detail_refinement: bool = False
    detail_max_components: int = 12
    detail_min_frequency: float = 12.0
    detail_min_improvement: float = 1e-7
    detail_hf_ratio_threshold: float = 0.85
    detail_base_sigma: float = 1.0
    detail_component_families: tuple[str, ...] = DEFAULT_DETAIL_COMPONENT_FAMILIES
    joint_amplitude_refit: bool = True
    amplitude_refit_interval: int = 2
    joint_parameter_refinement: bool = True
    parameter_refinement_passes: int = 1
    parameter_refinement_atom_limit: int = 8
    band_aware_candidates: bool = True
    def __post_init__(self) -> None:
        """Validate all fitting, objective, decomposition, and backend controls."""
        allowed = set(SUPPORTED_COMPONENT_FAMILIES)
        if self.max_components < 0 or self.max_iterations < 1 or self.fft_candidates < 1:
            raise ValueError("component/iteration/candidate counts are invalid")
        if (isinstance(self.spectral_noise_modes, bool)
                or not isinstance(self.spectral_noise_modes, int)
                or self.spectral_noise_modes < 1):
            raise ValueError("spectral_noise_modes must be a positive integer")
        if (isinstance(self.noise_seed_candidates, bool)
                or not isinstance(self.noise_seed_candidates, int)
                or self.noise_seed_candidates < 1):
            raise ValueError("noise_seed_candidates must be a positive integer")
        if self.fitting_resolution is not None and self.fitting_resolution < 8:
            raise ValueError("fitting_resolution must be at least 8 or None")
        if (not math.isfinite(self.min_frequency) or self.min_frequency < 0
                or (self.max_frequency is not None
                    and (not math.isfinite(self.max_frequency)
                         or self.max_frequency <= self.min_frequency))):
            raise ValueError("frequency bounds are invalid")
        if not math.isfinite(self.min_improvement) or self.min_improvement < 0:
            raise ValueError("min_improvement must be a finite, non-negative number")
        if not set(self.component_families) <= allowed:
            raise ValueError("unsupported component family")
        if (isinstance(self.band_workers, bool)
                or not isinstance(self.band_workers, int)
                or self.band_workers < 1):
            raise ValueError("band_workers must be a positive integer")
        if (isinstance(self.candidate_workers, bool)
                or not isinstance(self.candidate_workers, int)
                or self.candidate_workers < 1):
            raise ValueError("candidate_workers must be a positive integer")
        if self.compute_backend not in ("numpy", "cupy"):
            raise ValueError("compute_backend must be 'numpy' or 'cupy'")
        if (isinstance(self.gpu_batch_size, bool)
                or not isinstance(self.gpu_batch_size, int)
                or self.gpu_batch_size < 1):
            raise ValueError("gpu_batch_size must be a positive integer")
        if self.detail_max_components < 0:
            raise ValueError("detail_max_components must be non-negative")
        if (not math.isfinite(self.detail_min_frequency)
                or self.detail_min_frequency < 0):
            raise ValueError("detail_min_frequency must be finite and non-negative")
        if (self.detail_refinement and self.max_frequency is not None
                and self.detail_min_frequency >= self.max_frequency):
            raise ValueError("detail_min_frequency must be below max_frequency")
        if (not math.isfinite(self.detail_min_improvement)
                or self.detail_min_improvement < 0):
            raise ValueError("detail_min_improvement must be finite and non-negative")
        if (not math.isfinite(self.detail_hf_ratio_threshold)
                or self.detail_hf_ratio_threshold <= 0):
            raise ValueError("detail_hf_ratio_threshold must be finite and positive")
        if not math.isfinite(self.detail_base_sigma) or self.detail_base_sigma <= 0:
            raise ValueError("detail_base_sigma must be finite and positive")
        if not set(self.detail_component_families) <= allowed:
            raise ValueError("unsupported detail component family")
        if (isinstance(self.amplitude_refit_interval, bool)
                or not isinstance(self.amplitude_refit_interval, int)
                or self.amplitude_refit_interval < 1):
            raise ValueError("amplitude_refit_interval must be a positive integer")
        if not isinstance(self.joint_parameter_refinement, bool):
            raise ValueError("joint_parameter_refinement must be boolean")
        if (isinstance(self.parameter_refinement_passes, bool)
                or not isinstance(self.parameter_refinement_passes, int)
                or self.parameter_refinement_passes < 1):
            raise ValueError("parameter_refinement_passes must be a positive integer")
        if (isinstance(self.parameter_refinement_atom_limit, bool)
                or not isinstance(self.parameter_refinement_atom_limit, int)
                or self.parameter_refinement_atom_limit < 1):
            raise ValueError("parameter_refinement_atom_limit must be a positive integer")
        if not isinstance(self.band_aware_candidates, bool):
            raise ValueError("band_aware_candidates must be boolean")
        TextureLossWeights(self.spectrum_weight, self.histogram_weight,
                           self.autocorrelation_weight, self.gradient_weight,
                           self.mse_weight, self.local_structure_weight,
                           self.local_contrast_weight,
                           self.absolute_spectrum_weight,
                           self.oriented_spectrum_weight)
        for value, name in ((self.local_structure_scales, "local_structure_scales"),
                            (self.local_structure_orientations,
                             "local_structure_orientations"),
                            (self.local_structure_block_size,
                             "local_structure_block_size"),
                            (self.local_structure_candidate_limit,
                             "local_structure_candidate_limit")):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        create_decomposition(self.decomposition_method, self.decomposition_bands,
                             self.decomposition_base_sigma)

    @property
    def texture_loss_weights(self) -> TextureLossWeights:
        """Return manual weights and the full-image diagnostic weights."""
        return TextureLossWeights(self.spectrum_weight, self.histogram_weight,
                                  self.autocorrelation_weight, self.gradient_weight,
                                  self.mse_weight, self.local_structure_weight,
                                  self.local_contrast_weight,
                                  self.absolute_spectrum_weight,
                                  self.oriented_spectrum_weight)

@dataclass
class FitResult:
    """Fitted model plus full-resolution diagnostics."""
    model: ProceduralTextureModel
    metrics: dict[str, float]
    reconstruction: np.ndarray = field(repr=False)
    residual: np.ndarray = field(repr=False)
    metadata: dict = field(default_factory=dict)
    def evaluate(self, width: int, height: int) -> np.ndarray:
        """Evaluate the fitted model as a ``(height, width)`` float image."""
        return self.model.evaluate(width, height)
    def evaluate_region(
        self, width: int, height: int,
        u_bounds: tuple[float, float] = (0.0, 1.0),
        v_bounds: tuple[float, float] = (0.0, 1.0),
    ) -> np.ndarray:
        """Evaluate the fitted model outside its original UV domain."""
        return self.model.evaluate_region(width, height, u_bounds, v_bounds)
    def to_dict(self) -> dict:
        """Return the serializable model, metrics, and fitting metadata."""
        return {"schema_version": 1, "model": self.model.to_dict(), "metrics": self.metrics,
                "metadata": self.metadata}
    def save_json(self, path: str | Path) -> None:
        """Write the serializable fit result to a UTF-8 JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

class TextureFitter:
    """Fit scalar rasters to a sparse procedural model."""
    def __init__(self, config: FitConfig | None = None) -> None:
        """Create a fitter using validated ``config`` or the default controls."""
        self.config = config or FitConfig()
    def fit(self, image_array: np.ndarray, progress_callback: ProgressCallback | None = None,
            cancel_callback: CancelCallback | None = None) -> FitResult:
        """Normalize and fit an image, returning its model and diagnostics.

        Args:
            image_array: Grayscale, RGB, or RGBA raster accepted by
                :func:`normalize_image`.
            progress_callback: Optional ``(stage, fraction, message)`` reporter.
            cancel_callback: Optional predicate checked during long fitting work.

        Returns:
            A :class:`FitResult` containing the procedural model, reconstruction,
            signed residual, metrics, and fitting metadata.
        """
        target = normalize_image(np.asarray(image_array))
        model, metadata = fit_texture(target, self.config, progress_callback, cancel_callback)
        reconstruction = model.evaluate(target.shape[1], target.shape[0])
        metrics = calculate_metrics(target, reconstruction)
        metrics.update(calculate_texture_loss(
            target, reconstruction, self.config.texture_loss_weights,
            local_structure_scales=self.config.local_structure_scales,
            local_structure_orientations=self.config.local_structure_orientations,
            local_structure_block_size=self.config.local_structure_block_size))
        metadata["spectral_diagnostics"] = compare_spectra(target, reconstruction)
        return FitResult(model, metrics, reconstruction,
                         target - reconstruction, metadata)
