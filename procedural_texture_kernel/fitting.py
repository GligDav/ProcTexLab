"""Research-driven sparse, residual and multiscale fitting implementation."""
from __future__ import annotations
import time
from dataclasses import replace
from typing import TYPE_CHECKING
import numpy as np
from scipy.ndimage import zoom
from scipy.optimize import minimize
from .components import (AnisotropicGaussianComponent, BinaryPrimitiveComponent,
    DifferenceOfGaussiansComponent, DomainWarpedNoiseComponent,
    FractalBrownianMotionComponent, GaborComponent, GaussianRBFComponent, LineComponent,
    PerlinNoiseComponent, PolynomialTrendComponent, RadialWaveComponent,
    RidgedMultifractalComponent, SinusoidComponent, SparseImpulseComponent,
    SpiralWaveComponent, StepEdgeComponent, TurbulenceNoiseComponent,
    VoronoiNoiseComponent, WaveletComponent)
from .coordinates import coordinate_grid
from .model import ProceduralTextureModel
from .texture_loss import TextureLoss

if TYPE_CHECKING:
    from .api import FitConfig, ProgressCallback, CancelCallback

def _notify(callback, stage: str, progress: float, message: str) -> None:
    if callback is not None:
        callback(stage, float(np.clip(progress, 0, 1)), message)

def _resize_for_fit(image: np.ndarray, limit: int | None) -> np.ndarray:
    if limit is None or max(image.shape) <= limit:
        return image
    factor = limit / max(image.shape)
    shape = (max(8, round(image.shape[0] * factor)), max(8, round(image.shape[1] * factor)))
    return zoom(image, (shape[0] / image.shape[0], shape[1] / image.shape[1]), order=1)

def _solve_linear(model: ProceduralTextureModel, target: np.ndarray, u, v, ridge: float) -> None:
    """Initialize global DC/plane coefficients with stable least squares."""
    columns = [np.ones(target.size)]
    if model.trend_u != 0 or model.trend_v != 0:
        columns.extend([(u - .5).ravel(), (v - .5).ravel()])
    columns.extend(c.basis(u, v).ravel() for c in model.components)
    design = np.column_stack(columns)
    if ridge:
        design_aug = np.vstack([design, np.sqrt(ridge) * np.eye(design.shape[1])])
        target_aug = np.concatenate([target.ravel(), np.zeros(design.shape[1])])
    else:
        design_aug, target_aug = design, target.ravel()
    coefficients = np.linalg.lstsq(design_aug, target_aug, rcond=None)[0]
    model.bias = float(coefficients[0]); index = 1
    if model.trend_u != 0 or model.trend_v != 0:
        model.trend_u, model.trend_v = map(float, coefficients[index:index + 2]); index += 2
    for component, amplitude in zip(model.components, coefficients[index:]):
        component.amplitude = float(amplitude)

def _initial_plane(target, u, v, enabled: bool, ridge: float) -> ProceduralTextureModel:
    model = ProceduralTextureModel(trend_u=1.0 if enabled else 0.0,
                                   trend_v=1.0 if enabled else 0.0)
    _solve_linear(model, target, u, v, ridge)
    return model

def _fft_sinusoid_candidates(residual, config, count: int):
    """Use a Hann window to suppress non-periodic boundary leakage."""
    h, w = residual.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    spectrum = np.fft.fft2((residual - residual.mean()) * window)
    fy, fx = np.meshgrid(np.fft.fftfreq(h) * h, np.fft.fftfreq(w) * w, indexing="ij")
    radius = np.hypot(fx, fy)
    valid = (radius >= config.min_frequency) & (radius <= config.max_frequency)
    power = np.where(valid, np.abs(spectrum), -np.inf)
    order = np.argsort(power.ravel())[::-1]
    seen = set(); candidates = []
    for flat in order:
        if not np.isfinite(power.ravel()[flat]): break
        iy, ix = np.unravel_index(flat, power.shape)
        key = (abs(round(float(fx[iy, ix]), 8)), abs(round(float(fy[iy, ix]), 8)))
        if key in seen: continue
        seen.add(key)
        candidates.append(SinusoidComponent(frequency_u=float(fx[iy, ix]),
                                              frequency_v=float(fy[iy, ix])))
        if len(candidates) >= count: break
    return candidates

def _project(atom, residual, u, v):
    basis = atom.basis(u, v)
    denominator = float(np.vdot(basis, basis).real)
    amplitude = 0.0 if denominator < 1e-14 else float(np.vdot(residual, basis).real / denominator)
    return amplitude, float(amplitude * amplitude * denominator / residual.size)

def _phase_sinusoid(atom, residual, u, v):
    angle = 2 * np.pi * (atom.frequency_u * u + atom.frequency_v * v)
    design = np.column_stack([np.cos(angle).ravel(), np.sin(angle).ravel()])
    a, b = np.linalg.lstsq(design, residual.ravel(), rcond=None)[0]
    atom.amplitude = float(np.hypot(a, b)); atom.phase = float(np.arctan2(-b, a))
    return atom

def _local_candidates(residual, config, dominant_frequency: float):
    h, w = residual.shape; u, v = coordinate_grid(w, h)
    flat_order = np.argsort(np.abs(residual).ravel())[::-1]
    centers = []
    for flat in flat_order:
        iy, ix = np.unravel_index(flat, residual.shape)
        center = (float(u[iy, ix]), float(v[iy, ix]))
        if all((center[0]-a)**2 + (center[1]-b)**2 > .01 for a, b in centers):
            centers.append(center)
        if len(centers) == 4: break
    out = []
    for cu, cv in centers:
        if "gaussian_rbf" in config.component_families:
            out.extend(GaussianRBFComponent(center_u=cu, center_v=cv, sigma=s)
                       for s in (.06, .12, .22))
        if "gabor" in config.component_families and dominant_frequency >= 1:
            out.extend(GaborComponent(center_u=cu, center_v=cv, sigma_u=.16, sigma_v=.10,
                                      frequency=dominant_frequency, orientation=o)
                       for o in (0, np.pi/4, np.pi/2, 3*np.pi/4))
        if "wavelet" in config.component_families:
            out.extend(WaveletComponent(center_u=cu, center_v=cv, scale_u=s, scale_v=s)
                       for s in (.05, .10, .20))
        if "anisotropic_gaussian" in config.component_families:
            out.extend(AnisotropicGaussianComponent(center_u=cu, center_v=cv,
                       sigma_u=s, sigma_v=s/2, orientation=o)
                       for s in (.08, .16) for o in (0, np.pi/2))
        if "line" in config.component_families:
            out.extend(LineComponent(center_u=cu, center_v=cv, orientation=o)
                       for o in (0, np.pi/4, np.pi/2, 3*np.pi/4))
        if "step_edge" in config.component_families:
            out.extend(StepEdgeComponent(center_u=cu, center_v=cv, orientation=o)
                       for o in (0, np.pi/4, np.pi/2, 3*np.pi/4))
        if "dog_log" in config.component_families:
            out.extend(DifferenceOfGaussiansComponent(center_u=cu, center_v=cv,
                       sigma=s, mode=mode) for s in (.06, .12) for mode in ("dog", "log"))
        if "radial_wave" in config.component_families:
            out.append(RadialWaveComponent(center_u=cu, center_v=cv,
                                           frequency=max(1., dominant_frequency)))
        if "spiral_wave" in config.component_families:
            out.append(SpiralWaveComponent(center_u=cu, center_v=cv,
                                           frequency=max(1., dominant_frequency)))
        if "binary_primitive" in config.component_families:
            out.extend(BinaryPrimitiveComponent(center_u=cu, center_v=cv, shape=shape)
                       for shape in ("disk", "box", "ring"))
    if "polynomial_trend" in config.component_families:
        out.append(PolynomialTrendComponent())
    if "sparse_impulse" in config.component_families:
        out.append(SparseImpulseComponent(seed=config.seed))
    return out

def _perlin_candidates(config):
    frequencies = (2.0, 4.0, 8.0, 16.0)
    families = {"perlin_noise": PerlinNoiseComponent,
                "fbm": FractalBrownianMotionComponent,
                "ridged_multifractal": RidgedMultifractalComponent,
                "turbulence_noise": TurbulenceNoiseComponent,
                "domain_warped_noise": DomainWarpedNoiseComponent,
                "voronoi_noise": VoronoiNoiseComponent}
    return [cls(frequency=f, seed=config.seed + index)
            for family, cls in families.items() if family in config.component_families
            for index, f in enumerate(frequencies)
            if config.min_frequency <= f <= config.max_frequency]

def _refine_new_atom(atom, current, target_loss, u, v, max_iterations: int):
    """Refine one atom against the translation-tolerant composite texture loss."""
    if isinstance(atom, SinusoidComponent):
        x0 = [atom.amplitude, atom.frequency_u, atom.frequency_v, atom.phase]
        bound = max(1.0, np.hypot(atom.frequency_u, atom.frequency_v) * .35)
        bounds = [(-2, 2), (atom.frequency_u-bound, atom.frequency_u+bound),
                  (atom.frequency_v-bound, atom.frequency_v+bound), (-np.pi, np.pi)]
        def make(p): return SinusoidComponent(p[0], p[1], p[2], p[3])
    elif isinstance(atom, GaussianRBFComponent):
        x0 = [atom.amplitude, atom.center_u, atom.center_v, atom.sigma]
        bounds = [(-2, 2), (0, 1), (0, 1), (.015, .5)]
        def make(p): return GaussianRBFComponent(p[0], p[1], p[2], p[3])
    elif isinstance(atom, WaveletComponent):
        x0 = [atom.amplitude, atom.center_u, atom.center_v, atom.scale_u,
              atom.scale_v, atom.orientation]
        bounds = [(-2, 2), (0, 1), (0, 1), (.015, .5), (.015, .5),
                  (-np.pi, np.pi)]
        def make(p): return WaveletComponent(p[0], p[1], p[2], p[3], p[4], p[5])
    elif isinstance(atom, PerlinNoiseComponent):
        x0 = [atom.amplitude, atom.frequency, atom.offset_u, atom.offset_v]
        bounds = [(-2, 2), (.25, 32), (-1, 1), (-1, 1)]
        def make(p): return replace(atom, amplitude=p[0], frequency=p[1],
                                    offset_u=p[2], offset_v=p[3])
    elif isinstance(atom, GaborComponent):
        x0 = [atom.amplitude, atom.center_u, atom.center_v, atom.sigma_u, atom.sigma_v,
              atom.frequency, atom.orientation, atom.phase]
        bounds = [(-2, 2), (0, 1), (0, 1), (.02, .5), (.02, .5),
                  (.25, 32), (-np.pi, np.pi), (-np.pi, np.pi)]
        def make(p): return GaborComponent(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7])
    else:
        # These families have discrete modes or heterogeneous parameterizations;
        # projection already gives their exact least-squares amplitude.
        return atom
    def objective(p): return target_loss.evaluate(current + make(p).evaluate(u, v))[0]
    result = minimize(objective, x0, method="Nelder-Mead", bounds=bounds,
                      options={"maxiter": max_iterations, "xatol": 1e-5, "fatol": 1e-7})
    return make(result.x)

def fit_texture(target: np.ndarray, config: "FitConfig", progress_callback=None,
                cancel_callback=None) -> tuple[ProceduralTextureModel, dict]:
    """Fit using residual proposals and a composite statistical texture objective."""
    started = time.perf_counter(); fit_target = _resize_for_fit(target, config.fitting_resolution)
    h, w = fit_target.shape; u, v = coordinate_grid(w, h)
    _notify(progress_callback, "initialization", 0, "Estimating DC and planar trend")
    model = _initial_plane(fit_target, u, v, config.fit_plane, config.ridge)
    loss = TextureLoss(fit_target, config.texture_loss_weights)
    history = []
    for iteration in range(config.max_components):
        if cancel_callback is not None and cancel_callback():
            raise RuntimeError("fitting cancelled")
        current = model.evaluate_grid(u, v)
        residual = fit_target - current
        before, _ = loss.evaluate(current)
        candidates = []
        if "sinusoid" in config.component_families:
            candidates.extend(_fft_sinusoid_candidates(residual, config, config.fft_candidates))
        candidates.extend(_perlin_candidates(config))
        dominant = 0.0
        sinusoid_candidates = [x for x in candidates if isinstance(x, SinusoidComponent)]
        if sinusoid_candidates:
            dominant = float(np.hypot(sinusoid_candidates[0].frequency_u,
                                      sinusoid_candidates[0].frequency_v))
            for atom in sinusoid_candidates:
                _phase_sinusoid(atom, residual, u, v)
        candidates.extend(_local_candidates(residual, config, dominant))
        if not candidates: break
        scored = []
        for atom in candidates:
            if isinstance(atom, SinusoidComponent):
                score = float(np.mean((atom.amplitude * atom.basis(u, v))**2))
            else:
                atom.amplitude, score = _project(atom, residual, u, v)
            # Pixel correlation only initializes/shortlists atoms. Selection is
            # based on the phase/translation-tolerant texture objective.
            candidate_loss, _ = loss.evaluate(current + atom.evaluate(u, v))
            scored.append((before - candidate_loss, score, atom))
        improvement, _, chosen = max(scored, key=lambda item: (item[0], item[1]))
        if improvement <= config.min_improvement: break
        chosen = _refine_new_atom(chosen, current, loss, u, v, config.max_iterations)
        after, parts = loss.evaluate(current + chosen.evaluate(u, v))
        if before - after <= config.min_improvement:
            break
        model.add(chosen)
        history.append({"iteration": iteration + 1, "family": chosen.type_name,
                        "texture_loss": after, "improvement": before - after, **parts})
        _notify(progress_callback, "fitting", (iteration + 1) / max(config.max_components, 1),
                f"Added {chosen.type_name} atom {iteration + 1}/{config.max_components}")
    _notify(progress_callback, "complete", 1, "Fit complete")
    final_loss, final_parts = loss.evaluate(model.evaluate_grid(u, v))
    weights = config.texture_loss_weights
    return model, {"fit_shape": [h, w], "components": len(model.components),
                   "iterations": history, "elapsed_seconds": time.perf_counter() - started,
                   "seed": config.seed,
                   "objective": {"name": "composite_texture_loss", "final": final_loss,
                                 "components": final_parts,
                                 "weights": {"spectrum": weights.spectrum,
                                             "histogram": weights.histogram,
                                             "autocorrelation": weights.autocorrelation,
                                             "gradient": weights.gradient,
                                             "mse": weights.mse}}}
