"""Research-driven sparse, residual and multiscale fitting implementation."""
from __future__ import annotations
import time
from typing import TYPE_CHECKING, Callable
import numpy as np
from scipy.ndimage import zoom
from scipy.optimize import least_squares
from .components import GaborComponent, GaussianRBFComponent, SinusoidComponent
from .coordinates import coordinate_grid
from .model import ProceduralTextureModel

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
    """OMP amplitude refit using stable least squares, not normal equations."""
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

def _initial_plane(target, u, v, enabled: bool) -> ProceduralTextureModel:
    model = ProceduralTextureModel(trend_u=1.0 if enabled else 0.0,
                                   trend_v=1.0 if enabled else 0.0)
    _solve_linear(model, target, u, v, 0)
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
    return out

def _refine_new_atom(atom, residual, u, v, max_nfev: int):
    """Bounded variable projection: amplitude is eliminated at each nonlinear step."""
    if isinstance(atom, SinusoidComponent):
        x0 = [atom.frequency_u, atom.frequency_v]
        bound = max(1.0, np.hypot(*x0) * .35)
        lo, hi = [x0[0]-bound, x0[1]-bound], [x0[0]+bound, x0[1]+bound]
        def make(p): return SinusoidComponent(frequency_u=p[0], frequency_v=p[1])
    elif isinstance(atom, GaussianRBFComponent):
        x0 = [atom.center_u, atom.center_v, atom.sigma]
        lo, hi = [0, 0, .015], [1, 1, .5]
        def make(p): return GaussianRBFComponent(center_u=p[0], center_v=p[1], sigma=p[2])
    else:
        x0 = [atom.center_u, atom.center_v, atom.sigma_u, atom.sigma_v,
              atom.frequency, atom.orientation, atom.phase]
        lo, hi = [0, 0, .02, .02, .25, -np.pi, -np.pi], [1, 1, .5, .5, 32, np.pi, np.pi]
        def make(p): return GaborComponent(center_u=p[0], center_v=p[1], sigma_u=p[2],
                                           sigma_v=p[3], frequency=p[4], orientation=p[5], phase=p[6])
    def objective(p):
        candidate = make(p); amplitude, _ = _project(candidate, residual, u, v)
        return (residual - amplitude * candidate.basis(u, v)).ravel()
    result = least_squares(objective, x0, bounds=(lo, hi), max_nfev=max_nfev,
                           ftol=1e-7, xtol=1e-7, gtol=1e-7)
    refined = make(result.x)
    if isinstance(refined, SinusoidComponent): _phase_sinusoid(refined, residual, u, v)
    else: refined.amplitude, _ = _project(refined, residual, u, v)
    return refined

def fit_texture(target: np.ndarray, config: "FitConfig", progress_callback=None,
                cancel_callback=None) -> tuple[ProceduralTextureModel, dict]:
    """Fit using FFT candidates, residual pursuit, variable projection and OMP refits."""
    started = time.perf_counter(); fit_target = _resize_for_fit(target, config.fitting_resolution)
    h, w = fit_target.shape; u, v = coordinate_grid(w, h)
    _notify(progress_callback, "initialization", 0, "Estimating DC and planar trend")
    model = _initial_plane(fit_target, u, v, config.fit_plane)
    history = []
    for iteration in range(config.max_components):
        if cancel_callback is not None and cancel_callback():
            raise RuntimeError("fitting cancelled")
        residual = fit_target - model.evaluate_grid(u, v)
        before = float(np.mean(residual**2))
        candidates = []
        if "sinusoid" in config.component_families:
            candidates.extend(_fft_sinusoid_candidates(residual, config, config.fft_candidates))
        dominant = 0.0
        if candidates:
            dominant = float(np.hypot(candidates[0].frequency_u, candidates[0].frequency_v))
            candidates = [_phase_sinusoid(x, residual, u, v) for x in candidates]
        candidates.extend(_local_candidates(residual, config, dominant))
        if not candidates: break
        scored = []
        for atom in candidates:
            if isinstance(atom, SinusoidComponent):
                score = float(np.mean((atom.amplitude * atom.basis(u, v))**2))
            else:
                atom.amplitude, score = _project(atom, residual, u, v)
            scored.append((score, atom))
        score, chosen = max(scored, key=lambda item: item[0])
        if score <= config.min_improvement: break
        chosen = _refine_new_atom(chosen, residual, u, v, config.max_iterations)
        model.add(chosen); _solve_linear(model, fit_target, u, v, config.ridge)
        after = float(np.mean((fit_target - model.evaluate_grid(u, v))**2))
        if before - after <= config.min_improvement:
            model.components.pop(); _solve_linear(model, fit_target, u, v, config.ridge); break
        history.append({"iteration": iteration + 1, "family": chosen.type_name,
                        "mse": after, "improvement": before - after})
        _notify(progress_callback, "fitting", (iteration + 1) / max(config.max_components, 1),
                f"Added {chosen.type_name} atom {iteration + 1}/{config.max_components}")
    _notify(progress_callback, "complete", 1, "Fit complete")
    return model, {"fit_shape": [h, w], "components": len(model.components),
                   "iterations": history, "elapsed_seconds": time.perf_counter() - started,
                   "seed": config.seed}
