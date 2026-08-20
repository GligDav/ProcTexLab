"""Research-driven sparse, residual and multiscale fitting implementation."""
from __future__ import annotations
import time
from dataclasses import replace
from typing import TYPE_CHECKING
import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from scipy.optimize import minimize
from .components import (AnisotropicGaussianComponent, BinaryPrimitiveComponent,
    DifferenceOfGaussiansComponent, DomainWarpedNoiseComponent,
    FractalBrownianMotionComponent, GaborComponent, GaussianRBFComponent, LineComponent,
    MaskedNoiseComponent,
    PerlinNoiseComponent, PolynomialTrendComponent, RadialWaveComponent,
    RidgedMultifractalComponent, SinusoidComponent, SparseImpulseComponent,
    SpiralWaveComponent, StepEdgeComponent, ThresholdedNoiseComponent, TurbulenceNoiseComponent,
    VoronoiNoiseComponent, WaveletComponent, SimpleConstantComponent)
from .coordinates import coordinate_grid
from .decomposition import create_decomposition
from .model import ProceduralTextureModel
from .texture_loss import TextureLoss, TextureLossWeights
from .weight_estimator import WeightEstimator
from .spectral_diagnostics import compare_spectra
from .shader_graph import ShaderGraph, ShaderGraphComponent, ShaderNode

if TYPE_CHECKING:
    from .api import FitConfig, ProgressCallback, CancelCallback

_DETAIL_FAMILIES = frozenset({
    "sinusoid", "gabor", "wavelet", "dog_log", "sparse_impulse", "line",
    "perlin_noise", "turbulence_noise",
})
_STRUCTURE_FAMILIES = frozenset({
    "thresholded_noise", "domain_warped_noise", "ridged_multifractal",
    "masked_noise",
    "shader_graph",
    "voronoi_noise", "fbm", "anisotropic_gaussian", "gaussian_rbf",
    "step_edge", "polynomial_trend", "binary_primitive", "simple_constant",
    "radial_wave", "spiral_wave", "line",
})


def _families_for_band(config: "FitConfig", band_index: int,
                       band_count: int) -> tuple[str, ...]:
    """Select role-appropriate families; never suppress the sole available role."""
    selected = tuple(config.component_families)
    if not config.band_aware_candidates or band_count <= 1:
        return selected
    position = band_index / max(band_count - 1, 1)  # high frequency -> low residual
    preferred = (_DETAIL_FAMILIES if position < 1 / 3 else
                 _STRUCTURE_FAMILIES if position > 2 / 3 else
                 _DETAIL_FAMILIES | _STRUCTURE_FAMILIES)
    active = tuple(family for family in selected if family in preferred)
    return active or selected

def _notify(callback, stage: str, progress: float, message: str) -> None:
    if callback is not None:
        callback(stage, float(np.clip(progress, 0, 1)), message)

def _resize_for_fit(image: np.ndarray, limit: int | None) -> np.ndarray:
    if limit is None or max(image.shape) <= limit:
        return image
    factor = limit / max(image.shape)
    shape = (max(8, round(image.shape[0] * factor)), max(8, round(image.shape[1] * factor)))
    # Suppress frequencies that would alias into the smaller fitting raster.
    sigma = max(.5 / factor, .5)
    filtered = gaussian_filter(image, sigma=sigma, mode="reflect")
    return zoom(filtered, (shape[0] / image.shape[0], shape[1] / image.shape[1]),
                order=1, prefilter=False)

def _solve_linear(model: ProceduralTextureModel, target: np.ndarray, u, v, ridge: float,
                  include_plane: bool | None = None) -> None:
    """Initialize global DC/plane coefficients with stable least squares."""
    if include_plane is None:
        include_plane = model.trend_u != 0 or model.trend_v != 0
    columns = [np.ones(target.size)]
    if include_plane:
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
    if include_plane:
        model.trend_u, model.trend_v = map(float, coefficients[index:index + 2]); index += 2
    for component, amplitude in zip(model.components, coefficients[index:]):
        component.amplitude = float(amplitude)

def _initial_plane(target, u, v, enabled: bool, ridge: float) -> ProceduralTextureModel:
    model = ProceduralTextureModel(trend_u=1.0 if enabled else 0.0,
                                   trend_v=1.0 if enabled else 0.0)
    _solve_linear(model, target, u, v, ridge, include_plane=enabled)
    return model


def _refit_linear_amplitudes(model: ProceduralTextureModel, target: np.ndarray,
                             loss: TextureLoss, u, v, ridge: float,
                             include_plane: bool) -> dict:
    """Jointly refit all linear coefficients and retain only objective improvements."""
    before, _ = loss.evaluate(model.evaluate_grid(u, v))
    snapshot = (model.bias, model.trend_u, model.trend_v,
                [component.amplitude for component in model.components])
    _solve_linear(model, target, u, v, ridge, include_plane=include_plane)
    after, _ = loss.evaluate(model.evaluate_grid(u, v))
    accepted = after <= before + 1e-15
    if not accepted:
        model.bias, model.trend_u, model.trend_v = snapshot[:3]
        for component, amplitude in zip(model.components, snapshot[3]):
            component.amplitude = amplitude
        after = before
    return {"attempted": True, "accepted": accepted,
            "before": before, "after": after,
            "improvement": before - after}


def _refine_model_parameters(model: ProceduralTextureModel, target: np.ndarray,
                             loss: TextureLoss, u, v, config: "FitConfig",
                             progress_callback=None, cancel_callback=None) -> dict:
    """Revisit accepted atoms with coordinate-descent nonlinear refinement."""
    initial, _ = loss.evaluate(model.evaluate_grid(u, v))
    passes = []
    for pass_index in range(config.parameter_refinement_passes):
        pass_before, _ = loss.evaluate(model.evaluate_grid(u, v))
        accepted = 0
        first_index = max(0, len(model.components)
                          - config.parameter_refinement_atom_limit)
        for index in range(first_index, len(model.components)):
            atom = model.components[index]
            if cancel_callback is not None and cancel_callback():
                raise RuntimeError("fitting cancelled")
            full = model.evaluate_grid(u, v)
            current_without_atom = full - atom.evaluate(u, v)
            refined = _refine_new_atom(
                atom, current_without_atom, loss, u, v,
                max(8, config.max_iterations // 2), config.max_frequency)
            before, _ = loss.evaluate(full)
            after, _ = loss.evaluate(
                current_without_atom + refined.evaluate(u, v))
            if after < before - 1e-12:
                model.components[index] = refined
                accepted += 1
            completed = ((pass_index * (len(model.components) - first_index)
                          + index - first_index + 1)
                         / (config.parameter_refinement_passes
                            * max(len(model.components) - first_index, 1)))
            _notify(progress_callback, "parameter_refinement", completed,
                    f"Parameter pass {pass_index + 1}/"
                    f"{config.parameter_refinement_passes}: atom "
                    f"{index - first_index + 1}/"
                    f"{len(model.components) - first_index}")
        pass_after, _ = loss.evaluate(model.evaluate_grid(u, v))
        passes.append({"pass": pass_index + 1, "before": pass_before,
                       "after": pass_after, "improvement": pass_before - pass_after,
                       "accepted_atoms": accepted})
        if accepted == 0:
            break
    final, _ = loss.evaluate(model.evaluate_grid(u, v))
    return {"attempted": True, "passes": passes, "before": initial,
            "after": final, "improvement": initial - final,
            "accepted_atoms": sum(item["accepted_atoms"] for item in passes)}

def _fft_sinusoid_candidates(residual, config, count: int):
    """Use a Hann window to suppress non-periodic boundary leakage."""
    h, w = residual.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    spectrum = np.fft.fft2((residual - residual.mean()) * window)
    fy, fx = np.meshgrid(np.fft.fftfreq(h) * h, np.fft.fftfreq(w) * w, indexing="ij")
    radius = np.hypot(fx, fy)
    maximum = (config.max_frequency if config.max_frequency is not None
               else .45 * min(h, w))
    valid = (radius >= config.min_frequency) & (radius <= maximum)
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


def _local_structure_estimate(residual, iy: int, ix: int, gradients=None):
    """Estimate local edge normal, tangent, and support size around a residual peak."""
    if gradients is None:
        gradients = np.gradient(gaussian_filter(residual, 1.0))
    gy, gx = gradients
    radius = max(2, min(residual.shape) // 12)
    top, bottom = max(0, iy-radius), min(residual.shape[0], iy+radius+1)
    left, right = max(0, ix-radius), min(residual.shape[1], ix+radius+1)
    local_gx, local_gy = gx[top:bottom, left:right], gy[top:bottom, left:right]
    jxx = float(np.sum(local_gx * local_gx))
    jyy = float(np.sum(local_gy * local_gy))
    jxy = float(np.sum(local_gx * local_gy))
    normal = .5 * np.arctan2(2.0 * jxy, jxx - jyy) if jxx + jyy > 1e-15 else 0.0
    tangent = normal + np.pi / 2.0

    patch = np.abs(residual[top:bottom, left:right])
    yy, xx = np.indices(patch.shape)
    weights = patch / max(float(np.sum(patch)), 1e-12)
    spread_pixels = np.sqrt(float(np.sum(weights * (
        (xx - (ix-left)) ** 2 + (yy - (iy-top)) ** 2))) / 2.0)
    scale = float(np.clip(spread_pixels / max(residual.shape), .025, .25))
    return normal, tangent, scale


def _adaptive_noise_frequencies(residual, config):
    """Combine residual spectral peaks with stable octave anchors."""
    probes = _fft_sinusoid_candidates(residual, config, 2)
    proposed = [float(np.hypot(atom.frequency_u, atom.frequency_v)) for atom in probes]
    proposed.extend((2.0, 4.0, 8.0, 16.0))
    maximum = (config.max_frequency if config.max_frequency is not None
               else .45 * min(residual.shape))
    frequencies = []
    for value in proposed:
        value = float(np.clip(value, config.min_frequency, maximum))
        if config.min_frequency <= value <= maximum and all(
                abs(value-existing) > .25 for existing in frequencies):
            frequencies.append(value)
        if len(frequencies) == 3:
            break
    return tuple(frequencies)

def _local_candidates(residual, config, dominant_frequency: float):
    h, w = residual.shape; u, v = coordinate_grid(w, h)
    gradients = np.gradient(gaussian_filter(residual, 1.0))
    flat_order = np.argsort(np.abs(residual).ravel())[::-1]
    centers = []
    for flat in flat_order:
        iy, ix = np.unravel_index(flat, residual.shape)
        center = (float(u[iy, ix]), float(v[iy, ix]), iy, ix)
        if all((center[0]-a)**2 + (center[1]-b)**2 > .01
               for a, b, _, _ in centers):
            centers.append(center)
        if len(centers) == 4: break
    out = []
    for cu, cv, iy, ix in centers:
        normal, tangent, scale = _local_structure_estimate(
            residual, iy, ix, gradients)
        scales = tuple(float(np.clip(scale * factor, .02, .4))
                       for factor in (.65, 1.0, 1.6))
        if "gaussian_rbf" in config.component_families:
            out.extend(GaussianRBFComponent(center_u=cu, center_v=cv, sigma=s)
                       for s in scales)
        if "gabor" in config.component_families and dominant_frequency >= 1:
            out.extend(GaborComponent(center_u=cu, center_v=cv, sigma_u=.16, sigma_v=.10,
                                      frequency=dominant_frequency, orientation=o)
                       for o in (normal, tangent))
        if "wavelet" in config.component_families:
            out.extend(WaveletComponent(center_u=cu, center_v=cv, scale_u=s, scale_v=s)
                       for s in scales)
        if "anisotropic_gaussian" in config.component_families:
            out.extend(AnisotropicGaussianComponent(center_u=cu, center_v=cv,
                       sigma_u=s*1.6, sigma_v=s*.65, orientation=o)
                       for s in scales[:2] for o in (normal, tangent))
        if "line" in config.component_families:
            out.extend(LineComponent(center_u=cu, center_v=cv, width=max(.01, scale),
                                     orientation=o) for o in (tangent, tangent + np.pi/4))
        if "step_edge" in config.component_families:
            out.extend(StepEdgeComponent(center_u=cu, center_v=cv, orientation=o,
                                         softness=max(.005, scale/2))
                       for o in (normal, normal + np.pi/4))
        if "dog_log" in config.component_families:
            out.extend(DifferenceOfGaussiansComponent(center_u=cu, center_v=cv,
                       sigma=s, mode=mode) for s in scales[:2] for mode in ("dog", "log"))
        if "radial_wave" in config.component_families:
            out.append(RadialWaveComponent(center_u=cu, center_v=cv,
                                           frequency=max(1., dominant_frequency)))
        if "spiral_wave" in config.component_families:
            out.append(SpiralWaveComponent(center_u=cu, center_v=cv,
                                           frequency=max(1., dominant_frequency)))
        if "binary_primitive" in config.component_families:
            out.extend(BinaryPrimitiveComponent(center_u=cu, center_v=cv, shape=shape)
                       for shape in ("disk", "box", "ring"))
        if "simple_constant" in config.component_families:
            out.append(SimpleConstantComponent(value=float(residual[iy, ix])))
    if "polynomial_trend" in config.component_families:
        out.append(PolynomialTrendComponent())
    if "sparse_impulse" in config.component_families:
        out.append(SparseImpulseComponent(seed=config.seed))
    return out

def _perlin_candidates(residual, config, u, v):
    frequencies = _adaptive_noise_frequencies(residual, config)
    maximum = (config.max_frequency if config.max_frequency is not None
               else .45 * min(residual.shape))
    seeds = tuple(config.seed + index for index in range(config.noise_seed_candidates))
    families = {"perlin_noise": PerlinNoiseComponent,
                "fbm": FractalBrownianMotionComponent,
                "turbulence_noise": TurbulenceNoiseComponent,
                "domain_warped_noise": DomainWarpedNoiseComponent,
                "voronoi_noise": VoronoiNoiseComponent}
    candidates = [cls(frequency=f, seed=seed)
            for family, cls in families.items() if family in config.component_families
            for f in frequencies for seed in seeds]
    if "ridged_multifractal" in config.component_families:
        candidates.extend(
            RidgedMultifractalComponent(
                frequency=frequency, ridge_power=power,
                rotation=rotation, seed=seed)
            for frequency in frequencies for seed in seeds
            for power in (1.5, 3.0)
            for rotation in (0.0, np.pi / 4.0)
        )
    if "thresholded_noise" in config.component_families:
        positive_coverage = float(np.mean(residual > 0))
        coverages = tuple(np.unique(np.clip((positive_coverage, .35, .65), .05, .95)))
        for frequency in frequencies:
            for seed in seeds:
                prototype = ThresholdedNoiseComponent(frequency=frequency, seed=seed)
                noise = _perlin_basis_for_threshold(prototype, u, v)
                for threshold in np.quantile(noise, tuple(1.0-c for c in coverages)):
                    for edge_width in (.04, .12):
                        candidates.append(replace(
                            prototype, threshold=float(threshold), edge_width=edge_width))
    if "masked_noise" in config.component_families:
        coverage = float(np.clip(np.mean(residual > 0), .1, .9))
        for mask_frequency in frequencies:
            detail_frequency = float(np.clip(
                max(mask_frequency * 2.0, frequencies[0]),
                config.min_frequency, maximum))
            for seed in seeds:
                mask_source = ThresholdedNoiseComponent(
                    frequency=mask_frequency, seed=seed)
                noise = _perlin_basis_for_threshold(mask_source, u, v)
                threshold = float(np.quantile(noise, 1.0 - coverage))
                candidates.extend(MaskedNoiseComponent(
                    mask_frequency=mask_frequency, mask_threshold=threshold,
                    mask_seed=seed, detail_frequency=detail_frequency,
                    detail_seed=seed + 1009, invert_mask=invert)
                    for invert in (False, True))
    if "shader_graph" in config.component_families:
        coverage = float(np.clip(np.mean(residual > 0), .1, .9))
        for mask_frequency in frequencies:
            detail_frequency = float(np.clip(
                max(mask_frequency * 2.0, frequencies[0]),
                config.min_frequency, maximum))
            for seed in seeds:
                mask_source = PerlinNoiseComponent(
                    frequency=mask_frequency, octaves=4, seed=seed)
                mask_values = mask_source.basis(u, v)
                threshold = float(np.quantile(mask_values, 1.0-coverage))
                graph = ShaderGraph([
                    ShaderNode("mask_source", "component", component=mask_source),
                    ShaderNode("mask", "smoothstep", ("mask_source",),
                               edge0=threshold-.08, edge1=threshold+.08),
                    ShaderNode("detail_a", "component", component=PerlinNoiseComponent(
                        frequency=detail_frequency, octaves=3, seed=seed+1009)),
                    ShaderNode("detail_b", "component", component=PerlinNoiseComponent(
                        frequency=min(maximum, detail_frequency*1.5),
                        octaves=3, seed=seed+2017)),
                    ShaderNode("result", "mix", ("detail_a", "detail_b", "mask")),
                ], "result")
                candidates.append(ShaderGraphComponent(graph=graph))
    return candidates


def _perlin_basis_for_threshold(atom: ThresholdedNoiseComponent, u, v):
    """Evaluate the unremapped source field used to initialize thresholds."""
    du, dv = u - .5, v - .5
    c, s = np.cos(atom.rotation), np.sin(atom.rotation)
    ru = c * du + s * dv + .5
    rv = -s * du + c * dv + .5
    return PerlinNoiseComponent(
        frequency=atom.frequency, octaves=atom.octaves,
        persistence=atom.persistence, lacunarity=atom.lacunarity,
        offset_u=atom.offset_u, offset_v=atom.offset_v,
        seed=atom.seed).basis(ru, rv)

def _refine_new_atom(atom, current, target_loss, u, v, max_iterations: int,
                     max_frequency: float = 32.0):
    """Refine one atom against the translation-tolerant composite texture loss."""
    frequency_upper = max(float(max_frequency), .25)
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
    elif isinstance(atom, ThresholdedNoiseComponent):
        x0 = [atom.amplitude, atom.frequency, atom.offset_u, atom.offset_v,
              atom.rotation, atom.threshold, atom.edge_width]
        bounds = [(-2, 2), (.25, frequency_upper), (-1, 1), (-1, 1),
                  (-np.pi, np.pi), (-1, 1), (.005, .5)]
        def make(p): return replace(
            atom, amplitude=p[0], frequency=p[1], offset_u=p[2],
            offset_v=p[3], rotation=p[4], threshold=p[5], edge_width=p[6])
    elif isinstance(atom, MaskedNoiseComponent):
        x0 = [atom.amplitude, atom.mask_frequency, atom.mask_offset_u,
              atom.mask_offset_v, atom.mask_rotation, atom.mask_threshold,
              atom.mask_edge_width, atom.detail_frequency,
              atom.detail_offset_u, atom.detail_offset_v]
        bounds = [(-2, 2), (.25, frequency_upper), (-1, 1), (-1, 1),
                  (-np.pi, np.pi),
                  (-1, 1), (.005, .5), (.25, frequency_upper), (-1, 1), (-1, 1)]
        def make(p): return replace(
            atom, amplitude=p[0], mask_frequency=p[1], mask_offset_u=p[2],
            mask_offset_v=p[3], mask_rotation=p[4], mask_threshold=p[5],
            mask_edge_width=p[6], detail_frequency=p[7],
            detail_offset_u=p[8], detail_offset_v=p[9])
    elif isinstance(atom, RidgedMultifractalComponent):
        x0 = [atom.amplitude, atom.frequency, atom.offset_u, atom.offset_v,
              atom.ridge_offset, atom.ridge_power, atom.rotation, atom.anisotropy]
        bounds = [(-2, 2), (.25, frequency_upper), (-1, 1), (-1, 1),
                  (.25, 1.5), (.25, 8), (-np.pi, np.pi), (.25, 4)]
        def make(p): return replace(
            atom, amplitude=p[0], frequency=p[1], offset_u=p[2], offset_v=p[3],
            ridge_offset=p[4], ridge_power=p[5], rotation=p[6], anisotropy=p[7])
    elif isinstance(atom, DomainWarpedNoiseComponent):
        x0 = [atom.amplitude, atom.frequency, atom.offset_u, atom.offset_v,
              atom.warp_amplitude, atom.warp_frequency]
        bounds = [(-2, 2), (.25, frequency_upper), (-1, 1), (-1, 1),
                  (0, .75), (.25, min(16.0, frequency_upper))]
        def make(p): return replace(
            atom, amplitude=p[0], frequency=p[1], offset_u=p[2], offset_v=p[3],
            warp_amplitude=p[4], warp_frequency=p[5])
    elif isinstance(atom, (FractalBrownianMotionComponent, TurbulenceNoiseComponent)):
        x0 = [atom.amplitude, atom.frequency, atom.offset_u, atom.offset_v,
              atom.persistence, atom.lacunarity]
        bounds = [(-2, 2), (.25, frequency_upper), (-1, 1), (-1, 1),
                  (.1, .9), (1.25, 4)]
        def make(p): return replace(
            atom, amplitude=p[0], frequency=p[1], offset_u=p[2], offset_v=p[3],
            persistence=p[4], lacunarity=p[5])
    elif isinstance(atom, PerlinNoiseComponent):
        x0 = [atom.amplitude, atom.frequency, atom.offset_u, atom.offset_v]
        bounds = [(-2, 2), (.25, frequency_upper), (-1, 1), (-1, 1)]
        def make(p): return replace(atom, amplitude=p[0], frequency=p[1],
                                    offset_u=p[2], offset_v=p[3])
    elif isinstance(atom, GaborComponent):
        x0 = [atom.amplitude, atom.center_u, atom.center_v, atom.sigma_u, atom.sigma_v,
              atom.frequency, atom.orientation, atom.phase]
        bounds = [(-2, 2), (0, 1), (0, 1), (.02, .5), (.02, .5),
                  (.25, frequency_upper), (-np.pi, np.pi), (-np.pi, np.pi)]
        def make(p): return GaborComponent(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7])
    elif isinstance(atom, VoronoiNoiseComponent):
        x0 = [atom.amplitude, atom.frequency, atom.jitter, atom.offset_u, atom.offset_v]
        bounds = [(-2, 2), (.25, frequency_upper), (0, 1.5), (-1, 1), (-1, 1)]
        def make(p): return replace(
            atom, amplitude=p[0], frequency=p[1], jitter=p[2],
            offset_u=p[3], offset_v=p[4])
    elif isinstance(atom, AnisotropicGaussianComponent):
        x0 = [atom.amplitude, atom.center_u, atom.center_v, atom.sigma_u,
              atom.sigma_v, atom.orientation]
        bounds = [(-2, 2), (0, 1), (0, 1), (.01, .75), (.01, .75),
                  (-np.pi, np.pi)]
        def make(p): return replace(
            atom, amplitude=p[0], center_u=p[1], center_v=p[2],
            sigma_u=p[3], sigma_v=p[4], orientation=p[5])
    elif isinstance(atom, LineComponent):
        x0 = [atom.amplitude, atom.center_u, atom.center_v, atom.width,
              atom.length, atom.orientation, atom.softness]
        bounds = [(-2, 2), (0, 1), (0, 1), (.005, .5), (.02, 2),
                  (-np.pi, np.pi), (.002, .25)]
        def make(p): return replace(
            atom, amplitude=p[0], center_u=p[1], center_v=p[2], width=p[3],
            length=p[4], orientation=p[5], softness=p[6])
    elif isinstance(atom, StepEdgeComponent):
        x0 = [atom.amplitude, atom.center_u, atom.center_v,
              atom.orientation, atom.softness]
        bounds = [(-2, 2), (0, 1), (0, 1), (-np.pi, np.pi), (.001, .25)]
        def make(p): return replace(
            atom, amplitude=p[0], center_u=p[1], center_v=p[2],
            orientation=p[3], softness=p[4])
    elif isinstance(atom, DifferenceOfGaussiansComponent):
        x0 = [atom.amplitude, atom.center_u, atom.center_v, atom.sigma, atom.ratio]
        bounds = [(-2, 2), (0, 1), (0, 1), (.01, .5), (1.01, 4)]
        def make(p): return replace(
            atom, amplitude=p[0], center_u=p[1], center_v=p[2],
            sigma=p[3], ratio=p[4])
    elif isinstance(atom, ShaderGraphComponent):
        nodes = {node.node_id: node for node in atom.graph.nodes}
        required = {"mask_source", "mask", "detail_a", "detail_b", "result"}
        if set(nodes) != required:
            return atom
        mask_source = nodes["mask_source"].component
        detail_a = nodes["detail_a"].component
        detail_b = nodes["detail_b"].component
        if not all(isinstance(component, PerlinNoiseComponent)
                   for component in (mask_source, detail_a, detail_b)):
            return atom
        threshold = .5 * (nodes["mask"].edge0 + nodes["mask"].edge1)
        edge_width = .5 * (nodes["mask"].edge1 - nodes["mask"].edge0)
        x0 = [atom.amplitude, mask_source.frequency, mask_source.offset_u,
              mask_source.offset_v, threshold, edge_width,
              detail_a.frequency, detail_a.offset_u, detail_a.offset_v,
              detail_b.frequency, detail_b.offset_u, detail_b.offset_v]
        bounds = [(-2, 2), (.25, frequency_upper), (-1, 1), (-1, 1),
                  (-1, 1), (.005, .5),
                  (.25, frequency_upper), (-1, 1), (-1, 1),
                  (.25, frequency_upper), (-1, 1), (-1, 1)]
        def make(p):
            mask = replace(mask_source, frequency=p[1], offset_u=p[2], offset_v=p[3])
            first = replace(detail_a, frequency=p[6], offset_u=p[7], offset_v=p[8])
            second = replace(detail_b, frequency=p[9], offset_u=p[10], offset_v=p[11])
            graph = ShaderGraph([
                ShaderNode("mask_source", "component", component=mask),
                ShaderNode("mask", "smoothstep", ("mask_source",),
                           edge0=p[4]-p[5], edge1=p[4]+p[5]),
                ShaderNode("detail_a", "component", component=first),
                ShaderNode("detail_b", "component", component=second),
                ShaderNode("result", "mix", ("detail_a", "detail_b", "mask")),
            ], "result")
            return ShaderGraphComponent(p[0], graph)
    else:
        # These families have discrete modes or heterogeneous parameterizations;
        # projection already gives their exact least-squares amplitude.
        return atom
    def objective(p): return target_loss.evaluate(current + make(p).evaluate(u, v))[0]
    initial_loss = objective(x0)
    result = minimize(objective, x0, method="Nelder-Mead", bounds=bounds,
                      options={"maxiter": max_iterations, "xatol": 1e-5, "fatol": 1e-7})
    refined = make(result.x)
    return refined if np.isfinite(result.fun) and result.fun <= initial_loss else atom

def _fit_band(target: np.ndarray, config: "FitConfig", band_index: int,
              band_count: int, progress_callback=None,
              cancel_callback=None) -> tuple[ProceduralTextureModel, dict]:
    """Fit one target band without decomposing procedural candidates."""
    h, w = target.shape; u, v = coordinate_grid(w, h)
    active_families = _families_for_band(config, band_index, band_count)
    candidate_config = replace(config, component_families=active_families)
    model = _initial_plane(target, u, v, config.fit_plane, config.ridge)
    analysis = WeightEstimator().analyze(target) if config.adaptive_texture_weights else None
    if analysis is None:
        weights = config.texture_loss_weights
    else:
        estimated = analysis.weights
        weights = TextureLossWeights(estimated.spectrum, estimated.histogram,
                                     estimated.autocorrelation, estimated.gradient,
                                     config.mse_weight, config.local_structure_weight,
                                     config.local_contrast_weight,
                                     config.absolute_spectrum_weight,
                                     config.oriented_spectrum_weight)
    loss = TextureLoss(target, weights,
                       config.local_structure_scales,
                       config.local_structure_orientations,
                       config.local_structure_block_size)
    history = []
    for iteration in range(config.max_components):
        if cancel_callback is not None and cancel_callback():
            raise RuntimeError("fitting cancelled")
        current = model.evaluate_grid(u, v)
        residual = target - current
        before, _ = loss.evaluate(current)
        candidates = []
        if "sinusoid" in active_families:
            candidates.extend(_fft_sinusoid_candidates(
                residual, candidate_config, config.fft_candidates))
        candidates.extend(_perlin_candidates(residual, candidate_config, u, v))
        dominant = 0.0
        sinusoid_candidates = [x for x in candidates if isinstance(x, SinusoidComponent)]
        if sinusoid_candidates:
            dominant = float(np.hypot(sinusoid_candidates[0].frequency_u,
                                      sinusoid_candidates[0].frequency_v))
            for atom in sinusoid_candidates:
                _phase_sinusoid(atom, residual, u, v)
        elif any(family in active_families
                 for family in ("gabor", "radial_wave", "spiral_wave")):
            # Oriented/local carriers still need a residual-derived frequency
            # estimate when global sinusoid atoms themselves are disabled.
            frequency_probes = _fft_sinusoid_candidates(residual, candidate_config, 1)
            if frequency_probes:
                dominant = float(np.hypot(frequency_probes[0].frequency_u,
                                          frequency_probes[0].frequency_v))
        candidates.extend(_local_candidates(residual, candidate_config, dominant))
        if not candidates: break
        initialized = []
        for atom in candidates:
            if isinstance(atom, SinusoidComponent):
                score = float(np.mean((atom.amplitude * atom.basis(u, v))**2))
            else:
                atom.amplitude, score = _project(atom, residual, u, v)
            initialized.append((score, atom))
        if weights.local_structure > 0:
            initialized = sorted(initialized, key=lambda item: item[0], reverse=True)[
                :config.local_structure_candidate_limit]
        scored = []
        for score, atom in initialized:
            # Pixel correlation initializes and, for the expensive local
            # structure objective, shortlists atoms. Final selection still uses
            # the complete texture objective.
            candidate_loss, _ = loss.evaluate(current + atom.evaluate(u, v))
            scored.append((before - candidate_loss, score, atom))
        improvement, _, chosen = max(scored, key=lambda item: (item[0], item[1]))
        if improvement <= config.min_improvement: break
        chosen = _refine_new_atom(chosen, current, loss, u, v,
                                  config.max_iterations, config.max_frequency)
        after, parts = loss.evaluate(current + chosen.evaluate(u, v))
        if before - after <= config.min_improvement:
            break
        model.add(chosen)
        amplitude_refit = {"attempted": False, "accepted": False,
                           "before": after, "after": after, "improvement": 0.0}
        if (config.joint_amplitude_refit
                and (iteration + 1) % config.amplitude_refit_interval == 0):
            amplitude_refit = _refit_linear_amplitudes(
                model, target, loss, u, v, config.ridge, config.fit_plane)
            after, parts = loss.evaluate(model.evaluate_grid(u, v))
        history.append({"band": band_index + 1, "iteration": iteration + 1,
                        "family": chosen.type_name,
                        "texture_loss": after, "improvement": before - after,
                        "amplitude_refit": amplitude_refit, **parts})
        completed = band_index * config.max_components + iteration + 1
        total = max(band_count * config.max_components, 1)
        _notify(progress_callback, "fitting", completed / total,
                f"Band {band_index + 1}/{band_count}: added {chosen.type_name} "
                f"atom {iteration + 1}/{config.max_components}")
    final_refit = {"attempted": False, "accepted": False}
    if (config.joint_amplitude_refit and model.components
            and len(model.components) % config.amplitude_refit_interval != 0):
        final_refit = _refit_linear_amplitudes(
            model, target, loss, u, v, config.ridge, config.fit_plane)
    parameter_refinement = {"attempted": False, "passes": [],
                            "accepted_atoms": 0}
    if config.joint_parameter_refinement and model.components:
        parameter_refinement = _refine_model_parameters(
            model, target, loss, u, v, config,
            progress_callback, cancel_callback)
        if config.joint_amplitude_refit:
            final_refit = _refit_linear_amplitudes(
                model, target, loss, u, v, config.ridge, config.fit_plane)
    final_loss, final_parts = loss.evaluate(model.evaluate_grid(u, v))
    result = {"band": band_index + 1, "components": len(model.components),
                   "candidate_families": list(active_families),
                   "iterations": history, "final_loss": final_loss,
                   "final_amplitude_refit": final_refit,
                   "parameter_refinement": parameter_refinement,
                   "loss_components": final_parts,
                   "weights": {"spectrum": weights.spectrum,
                               "histogram": weights.histogram,
                               "autocorrelation": weights.autocorrelation,
                               "gradient": weights.gradient,
                                "mse": weights.mse,
                               "local_structure": weights.local_structure,
                               "local_contrast": weights.local_contrast,
                               "absolute_spectrum": weights.absolute_spectrum}}
    result["weights"]["oriented_spectrum"] = weights.oriented_spectrum
    if analysis is not None:
        result["features"] = analysis.features.to_dict()
    return model, result


def _combine_models(models: list[ProceduralTextureModel]) -> ProceduralTextureModel:
    """Add independently fitted band models into one serializable model."""
    return ProceduralTextureModel(
        bias=sum(model.bias for model in models),
        trend_u=sum(model.trend_u for model in models),
        trend_v=sum(model.trend_v for model in models),
        components=[component for model in models for component in model.components],
    )


def _high_frequency_energy(diagnostics: dict, side: str) -> float:
    return float(sum(band[f"{side}_energy"] for band in diagnostics["bands"]
                     if band["name"] in ("high", "very_high")))


def _refine_high_frequency(target: np.ndarray, model: ProceduralTextureModel,
                           config: "FitConfig", progress_callback=None,
                           cancel_callback=None) -> tuple[ProceduralTextureModel, dict]:
    """Fit the high-pass reconstruction residual when diagnostics show a deficit."""
    h, w = target.shape
    before_image = model.evaluate(w, h)
    before_diagnostics = compare_spectra(target, before_image)
    metadata = {"enabled": True, "attempted": False, "accepted": False,
                "threshold": config.detail_hf_ratio_threshold,
                "before": before_diagnostics,
                "components": 0}
    if before_diagnostics["high_frequency_ratio"] >= config.detail_hf_ratio_threshold:
        metadata["reason"] = "high_frequency_ratio_meets_threshold"
        return model, metadata
    families = tuple(family for family in config.detail_component_families
                     if family in config.component_families)
    if config.detail_max_components == 0:
        metadata["reason"] = "zero_detail_component_budget"
        return model, metadata
    if not families:
        metadata["reason"] = "no_enabled_detail_component_families"
        return model, metadata
    if cancel_callback is not None and cancel_callback():
        raise RuntimeError("fitting cancelled")

    residual = target - before_image
    detail_target = residual - gaussian_filter(
        residual, config.detail_base_sigma, mode="reflect")
    detail_config = replace(
        config, max_components=config.detail_max_components,
        min_frequency=min(max(config.min_frequency, config.detail_min_frequency),
                          config.max_frequency * .75),
        min_improvement=config.detail_min_improvement,
        component_families=families, fit_plane=False,
        adaptive_texture_weights=False, spectrum_weight=.25,
        histogram_weight=0.0, autocorrelation_weight=.25,
        gradient_weight=1.0, mse_weight=4.0,
        detail_refinement=False,
    )
    metadata["attempted"] = True
    metadata["component_families"] = list(families)
    metadata["residual_rms"] = float(np.sqrt(np.mean(residual * residual)))
    metadata["detail_target_rms"] = float(np.sqrt(np.mean(detail_target * detail_target)))
    _notify(progress_callback, "detail_refinement", 0,
            "Fitting high-frequency reconstruction residual")
    detail_model, detail_result = _fit_band(
        detail_target, detail_config, 0, 1, progress_callback, cancel_callback)
    candidate_model = _combine_models([model, detail_model])
    after_image = candidate_model.evaluate(w, h)
    after_diagnostics = compare_spectra(target, after_image)
    target_hf = _high_frequency_energy(before_diagnostics, "target")
    before_error = abs(_high_frequency_energy(before_diagnostics, "result") - target_hf)
    after_error = abs(_high_frequency_energy(after_diagnostics, "result") - target_hf)
    before_mse = float(np.mean((target - before_image) ** 2))
    after_mse = float(np.mean((target - after_image) ** 2))
    accepted = (len(detail_model.components) > 0 and after_error < before_error
                and after_mse <= before_mse + 1e-12)
    metadata.update({"accepted": accepted, "after": after_diagnostics,
                     "components": len(detail_model.components),
                     "iterations": detail_result["iterations"],
                     "loss_components": detail_result["loss_components"],
                     "before_hf_absolute_error": before_error,
                     "after_hf_absolute_error": after_error,
                     "before_mse": before_mse, "after_mse": after_mse})
    if not accepted:
        metadata["reason"] = "candidate_did_not_improve_hf_energy_and_mse"
        return model, metadata
    _notify(progress_callback, "detail_refinement", 1,
            f"Accepted {len(detail_model.components)} high-frequency detail atoms")
    return candidate_model, metadata


def fit_texture(target: np.ndarray, config: "FitConfig", progress_callback=None,
                cancel_callback=None) -> tuple[ProceduralTextureModel, dict]:
    """Decompose the target, fit every band independently, then add the models."""
    started = time.perf_counter(); fit_target = _resize_for_fit(target, config.fitting_resolution)
    h, w = fit_target.shape
    automatic_frequency = config.max_frequency is None
    if automatic_frequency:
        # Preserve an anti-aliasing margin while allowing substantially more
        # detail than the former fixed 24-cycle ceiling.
        auto_max_frequency = max(config.min_frequency + .25, .45 * min(h, w))
        config = replace(config, max_frequency=auto_max_frequency)
    decomposition = create_decomposition(config.decomposition_method,
                                         config.decomposition_bands,
                                         config.decomposition_base_sigma)
    target_bands = decomposition.decompose(fit_target)
    models = []
    band_results = []
    for band_index, band_target in enumerate(target_bands):
        _notify(progress_callback, "initialization",
                band_index / max(len(target_bands), 1),
                f"Initializing band {band_index + 1}/{len(target_bands)}")
        band_model, band_result = _fit_band(
            band_target, config, band_index, len(target_bands),
            progress_callback, cancel_callback)
        models.append(band_model); band_results.append(band_result)
    model = _combine_models(models)
    if config.detail_refinement:
        model, detail_result = _refine_high_frequency(
            fit_target, model, config, progress_callback, cancel_callback)
    else:
        detail_result = {"enabled": False, "attempted": False, "accepted": False,
                         "reason": "disabled", "components": 0}
    _notify(progress_callback, "complete", 1, "Fit complete")
    band_losses = [result["final_loss"] for result in band_results]
    history = [item for result in band_results for item in result["iterations"]]
    return model, {"fit_shape": [h, w], "components": len(model.components),
                   "iterations": history, "elapsed_seconds": time.perf_counter() - started,
                   "seed": config.seed,
                   "noise_seed_candidates": config.noise_seed_candidates,
                   "frequency_range": {"minimum": config.min_frequency,
                                       "maximum": config.max_frequency,
                                       "maximum_mode": ("automatic" if
                                           automatic_frequency else "explicit")},
                   "decomposition": {"method": config.decomposition_method,
                                     "bands": config.decomposition_bands,
                                     "base_sigma": config.decomposition_base_sigma,
                                     "sigmas": list(getattr(decomposition, "sigmas", ()))},
                   "bands": band_results,
                   "detail_refinement": detail_result,
                   "objective": {"name": "independent_band_texture_loss",
                                 "final": float(np.mean(band_losses)),
                                 "band_losses": band_losses,
                                 "weight_mode": ("adaptive_per_band" if
                                    config.adaptive_texture_weights else "manual"),
                                 "band_aware_candidates": config.band_aware_candidates,
                                 "mse_weight": config.mse_weight,
                                 "absolute_spectrum_weight":
                                     config.absolute_spectrum_weight,
                                 "oriented_spectrum_weight":
                                     config.oriented_spectrum_weight,
                                 "joint_parameter_refinement":
                                     config.joint_parameter_refinement,
                                 "parameter_refinement_passes":
                                     config.parameter_refinement_passes,
                                 "parameter_refinement_atom_limit":
                                     config.parameter_refinement_atom_limit,
                                 "local_contrast_weight": config.local_contrast_weight,
                                 "local_structure": {
                                     "weight": config.local_structure_weight,
                                     "scales": config.local_structure_scales,
                                     "orientations": config.local_structure_orientations,
                                     "block_size": config.local_structure_block_size,
                                     "candidate_limit": config.local_structure_candidate_limit,
                                 },
                                 "band_weights": [result["weights"]
                                                  for result in band_results]}}
