"""Optional CuPy candidate batching for high-resolution fitting."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable
import numpy as np
from scipy.ndimage import gaussian_filter as numpy_gaussian_filter
from scipy.ndimage import zoom as numpy_zoom

from .components import (
    AnisotropicGaussianComponent, DifferenceOfGaussiansComponent,
    GaborComponent, GaussianRBFComponent, LineComponent,
    PolynomialTrendComponent, RadialWaveComponent, SimpleConstantComponent,
    SinusoidComponent, SpectralNoiseComponent, SpiralWaveComponent,
    StepEdgeComponent, WaveletComponent,
)


GPU_COMPONENT_TYPES = (
    SinusoidComponent, SpectralNoiseComponent, GaborComponent,
    GaussianRBFComponent, WaveletComponent, AnisotropicGaussianComponent,
    LineComponent, StepEdgeComponent, DifferenceOfGaussiansComponent,
    PolynomialTrendComponent, RadialWaveComponent, SpiralWaveComponent,
    SimpleConstantComponent,
)


def load_cupy():
    """Import CuPy lazily and verify that a CUDA device is available."""
    try:
        import cupy as cp
        from cupyx.scipy.ndimage import gaussian_filter
    except ImportError as exc:
        raise RuntimeError(
            "compute_backend='cupy' requires an optional CuPy package matching "
            "the installed CUDA runtime; install the project's gpu extra") from exc
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("CuPy is installed but no CUDA device is available")
        # Device enumeration alone can succeed with a broken runtime/compiler
        # setup.  Force a tiny allocation and kernel launch so selecting CuPy
        # fails here, with a useful error, rather than halfway through a fit.
        probe = cp.asarray((1.0,), dtype=cp.float64)
        cp.sum(probe).get()
    except Exception as exc:
        if isinstance(exc, RuntimeError) and str(exc).startswith("CuPy is installed"):
            raise
        raise RuntimeError("CuPy could not initialize a CUDA device") from exc
    return cp, gaussian_filter


@dataclass(frozen=True)
class NumericBackend:
    """NumPy/SciPy-compatible operations selected for one fit.

    Public models and results deliberately remain NumPy objects.  ``to_numpy``
    marks the few host/device boundaries used by preprocessing and proposal
    analysis; candidate batches remain resident in :class:`CuPyCandidateScorer`.
    """

    name: str
    xp: object
    gaussian_filter: object
    zoom: object

    @property
    def accelerated(self) -> bool:
        return self.name == "cupy"

    def to_numpy(self, value):
        return self.xp.asnumpy(value) if self.accelerated else np.asarray(value)


def numeric_backend(name: str) -> NumericBackend:
    """Initialize the requested numerical backend.

    CuPy is intentionally loaded only when explicitly requested.  This keeps
    the base installation and the GUI's NumPy selection CUDA-independent.
    """
    if name == "numpy":
        return NumericBackend("numpy", np, numpy_gaussian_filter, numpy_zoom)
    if name != "cupy":
        raise ValueError("compute backend must be 'numpy' or 'cupy'")
    cp, gaussian_filter = load_cupy()
    try:
        from cupyx.scipy.ndimage import zoom
    except ImportError as exc:
        raise RuntimeError("the installed CuPy package does not provide cupyx.scipy.ndimage") from exc
    return NumericBackend("cupy", cp, gaussian_filter, zoom)


def _basis_batch(atoms, u, v, cp):
    """Evaluate one homogeneous supported family as ``(N, H, W)``."""
    first = atoms[0]
    shape = (len(atoms), 1, 1)
    p = lambda name: cp.asarray([getattr(a, name) for a in atoms], dtype=cp.float64).reshape(shape)
    uu, vv = u[None, :, :], v[None, :, :]
    if isinstance(first, SinusoidComponent):
        return cp.cos(2 * cp.pi * (p("frequency_u") * uu + p("frequency_v") * vv)
                      + p("phase"))
    if isinstance(first, SpectralNoiseComponent):
        result = cp.zeros((len(atoms), *u.shape), dtype=cp.float64)
        for index, atom in enumerate(atoms):
            for fu, fv, weight, phase in zip(atom.frequencies_u, atom.frequencies_v,
                                              atom.weights, atom.phases):
                result[index] += weight * cp.cos(2 * cp.pi * (fu*u + fv*v) + phase)
            rms = np.sqrt(.5 * np.sum(np.asarray(atom.weights, dtype=float) ** 2))
            result[index] /= max(float(rms), 1e-12)
        return result
    if isinstance(first, (GaborComponent, WaveletComponent,
                          AnisotropicGaussianComponent, LineComponent)):
        du, dv = uu - p("center_u"), vv - p("center_v")
        c, s = cp.cos(p("orientation")), cp.sin(p("orientation"))
        x, y = c*du + s*dv, -s*du + c*dv
        if isinstance(first, GaborComponent):
            envelope = cp.exp(-.5*((x/p("sigma_u"))**2 + (y/p("sigma_v"))**2))
            return envelope * cp.cos(2*cp.pi*p("frequency")*x + p("phase"))
        if isinstance(first, WaveletComponent):
            r2 = (x/p("scale_u"))**2 + (y/p("scale_v"))**2
            return (1-r2) * cp.exp(-.5*r2)
        if isinstance(first, AnisotropicGaussianComponent):
            return cp.exp(-.5*((x/p("sigma_u"))**2 + (y/p("sigma_v"))**2))
        d = cp.maximum(cp.abs(y)-p("width")/2, cp.abs(x)-p("length")/2)
        return 1/(1+cp.exp(cp.clip(d/cp.maximum(p("softness"), 1e-12), -60, 60)))
    if isinstance(first, GaussianRBFComponent):
        r2 = (uu-p("center_u"))**2 + (vv-p("center_v"))**2
        return cp.exp(-.5*r2/p("sigma")**2)
    if isinstance(first, StepEdgeComponent):
        du, dv = uu-p("center_u"), vv-p("center_v")
        x = cp.cos(p("orientation"))*du + cp.sin(p("orientation"))*dv
        softness = p("softness")
        return cp.where(softness <= 0, cp.where(x >= 0, 1., -1.),
                        cp.tanh(x/cp.maximum(softness, 1e-12)))
    if isinstance(first, DifferenceOfGaussiansComponent):
        r2 = (uu-p("center_u"))**2 + (vv-p("center_v"))**2
        q = r2/p("sigma")**2
        dog = cp.exp(-.5*q) - cp.exp(-.5*r2/(cp.maximum(p("ratio"), 1.000001)
                                                   * p("sigma"))**2)/p("ratio")**2
        log = (1-.5*q)*cp.exp(-.5*q)
        modes = cp.asarray([a.mode == "log" for a in atoms]).reshape(shape)
        return cp.where(modes, log, dog)
    if isinstance(first, PolynomialTrendComponent):
        x, y = uu-.5, vv-.5
        return (p("linear_u")*x + p("linear_v")*y + p("quadratic_u")*x*x
                + p("cross_uv")*x*y + p("quadratic_v")*y*y)
    if isinstance(first, (RadialWaveComponent, SpiralWaveComponent)):
        du, dv = uu-p("center_u"), vv-p("center_v")
        radius = cp.hypot(du, dv)
        phase = 2*cp.pi*p("frequency")*radius + p("phase")
        if isinstance(first, SpiralWaveComponent):
            phase += p("arms")*cp.arctan2(dv, du)
        return cp.cos(phase)*cp.exp(-p("decay")*radius)
    if isinstance(first, SimpleConstantComponent):
        return cp.broadcast_to(p("value"), (len(atoms), *u.shape)).copy()
    raise TypeError(type(first).__name__)


class CuPyCandidateScorer:
    """Keep per-band arrays on CUDA and score homogeneous candidate batches."""
    def __init__(self, target_loss, u, v, batch_size: int):
        self.cp, self.gaussian_filter = load_cupy()
        self.loss = target_loss
        self.u = self.cp.asarray(u)
        self.v = self.cp.asarray(v)
        self.reference = self.cp.asarray(target_loss.reference)
        self.batch_size = batch_size

    def supported(self, atom) -> bool:
        # The local-structure filter bank has no GPU implementation yet.
        return isinstance(atom, GPU_COMPONENT_TYPES) and not self.loss.weights.local_structure

    def _feature_loss(self, images):
        """Mirror enabled TextureLoss terms on a resident candidate batch."""
        cp, weights = self.cp, self.loss.weights
        count = images.shape[0]
        totals = cp.zeros(count, dtype=cp.float64)
        denominator = sum((weights.spectrum, weights.histogram, weights.autocorrelation,
                           weights.gradient, weights.mse, weights.local_contrast,
                           weights.absolute_spectrum, weights.oriented_spectrum))
        if weights.mse:
            totals += weights.mse * cp.mean((images-self.reference)**2, axis=(1, 2))
        if weights.spectrum:
            current = images
            values = cp.zeros(count, dtype=cp.float64)
            for reference_feature in self.loss._spectrum:
                centered = current-cp.mean(current, axis=(1, 2), keepdims=True)
                power = cp.abs(cp.fft.rfft2(centered, axes=(-2, -1)))**2
                power /= cp.maximum(cp.sum(power, axis=(1, 2), keepdims=True), 1e-12)
                feature = cp.log1p(power*power.shape[-2]*power.shape[-1])
                values += cp.mean((feature-cp.asarray(reference_feature))**2, axis=(1, 2))
                if min(current.shape[-2:]) < 16:
                    break
                current = self.gaussian_filter(current, (0, 1., 1.))[..., ::2, ::2]
            totals += weights.spectrum * values / len(self.loss._spectrum)
        if weights.absolute_spectrum or weights.oriented_spectrum:
            height, width = images.shape[-2:]
            window = cp.outer(cp.hanning(height), cp.hanning(width))
            window_power = cp.maximum(cp.mean(window*window), cp.finfo(cp.float64).tiny)
            transformed = cp.fft.fft2(
                (images-cp.mean(images, axis=(1, 2), keepdims=True))*window,
                axes=(-2, -1))
            power = cp.abs(transformed)**2/(height*width)**2/window_power
            fy = cp.fft.fftfreq(height)[:, None]
            fx = cp.fft.fftfreq(width)[None, :]
            radius = cp.hypot(fy, fx)/.5
            edges = (0., .125, .25, .5, .75, np.sqrt(2.)+1e-12)
            absolute = cp.stack([cp.sum(power[:, (radius >= lo) & (radius < hi)], axis=1)
                                 for lo, hi in zip(edges[:-1], edges[1:])], axis=1)
            if weights.absolute_spectrum:
                ratio = cp.log10((absolute+self.loss._absolute_spectrum_epsilon)/(
                    cp.asarray(self.loss._absolute_spectrum)
                    + self.loss._absolute_spectrum_epsilon))/8.
                totals += weights.absolute_spectrum*cp.mean(ratio*ratio, axis=1)
            if weights.oriented_spectrum:
                orientations = self.loss._oriented_spectrum.shape[1]
                angle = cp.mod(cp.arctan2(fy, fx), cp.pi)
                wedge = cp.minimum((angle*orientations/cp.pi).astype(cp.int64),
                                   orientations-1)
                oriented = cp.stack([
                    cp.stack([cp.sum(power[:, (radius >= lo) & (radius < hi)
                                                & (wedge == direction)], axis=1)
                              for direction in range(orientations)], axis=1)
                    for lo, hi in zip(edges[:-1], edges[1:])], axis=1)
                ratio = cp.log10((oriented+self.loss._oriented_spectrum_epsilon)/(
                    cp.asarray(self.loss._oriented_spectrum)
                    + self.loss._oriented_spectrum_epsilon))/8.
                totals += weights.oriented_spectrum*cp.mean(ratio*ratio, axis=(1, 2))
        # Less common/non-batch-friendly features are evaluated from resident
        # candidate arrays one at a time; no image crosses PCIe during scoring.
        for index in range(count):
            image = images[index]
            if weights.histogram:
                histogram, _ = cp.histogram(cp.clip(image, *self.loss._histogram_range),
                                             bins=64, range=self.loss._histogram_range)
                cdf = cp.cumsum(histogram, dtype=cp.float64)/image.size
                totals[index] += weights.histogram*cp.mean(cp.abs(
                    cp.asarray(self.loss._histogram)-cdf))
            if weights.autocorrelation:
                centered = image-cp.mean(image); variance = cp.mean(centered*centered)
                if float(variance.get()) < 1e-12:
                    feature = cp.zeros_like(cp.asarray(self.loss._autocorrelation))
                else:
                    correlation = cp.fft.fftshift(cp.fft.ifft2(
                        cp.abs(cp.fft.fft2(centered))**2).real/(image.size*variance))
                    ry = min(8, image.shape[0]//2); rx = min(8, image.shape[1]//2)
                    cy, cx = image.shape[0]//2, image.shape[1]//2
                    feature = correlation[cy-ry:cy+ry+1, cx-rx:cx+rx+1]
                totals[index] += weights.autocorrelation*cp.mean((
                    cp.asarray(self.loss._autocorrelation)-feature)**2)
            if weights.gradient:
                features = []
                for scale_index, sigma in enumerate((0., 1., 2., 4.)):
                    working = image if sigma == 0 else self.gaussian_filter(image, sigma)
                    dx = cp.roll(working, -1, axis=1)-working
                    dy = cp.roll(working, -1, axis=0)-working
                    magnitude = cp.hypot(dx, dy)
                    scale = self.loss._gradient_normalizers[scale_index]
                    magnitude_hist, _ = cp.histogram(cp.clip(magnitude/scale, 0, 1),
                                                     bins=32, range=(0, 1))
                    angles = cp.mod(cp.arctan2(dy, dx), cp.pi)
                    orientation_hist, _ = cp.histogram(
                        angles, bins=16, range=(0, cp.pi), weights=magnitude)
                    orientation_hist = orientation_hist.astype(cp.float64)
                    orientation_hist /= cp.maximum(cp.sum(orientation_hist), 1e-12)
                    features.extend((cp.mean(magnitude)[None], cp.std(magnitude)[None],
                                     magnitude_hist.astype(cp.float64)/image.size,
                                     orientation_hist))
                feature = cp.concatenate(features)
                totals[index] += weights.gradient*cp.mean(cp.abs(
                    cp.asarray(self.loss._gradient)-feature))
            if weights.local_contrast:
                features = []; squared = image**2
                for scale_index, sigma in enumerate((1., 2., 4., 8.)):
                    mean = self.gaussian_filter(image, sigma)
                    variance = cp.maximum(self.gaussian_filter(squared, sigma)-mean*mean, 0.)
                    contrast = cp.sqrt(variance)
                    scale = self.loss._contrast_normalizers[scale_index]
                    histogram, _ = cp.histogram(cp.clip(contrast/scale, 0, 1),
                                                bins=32, range=(0, 1))
                    features.extend((cp.mean(contrast)[None], cp.std(contrast)[None],
                                     histogram.astype(cp.float64)/image.size))
                feature = cp.concatenate(features)
                totals[index] += weights.local_contrast*cp.mean(cp.abs(
                    cp.asarray(self.loss._local_contrast)-feature))
        return totals/denominator

    def can_score_complete_loss(self) -> bool:
        return not self.loss.weights.local_structure

    def prepare_iteration(self, current, residual) -> None:
        """Transfer the two changing band arrays once for all family batches."""
        self.current = self.cp.asarray(current)
        self.residual = self.cp.asarray(residual)

    def project_and_score(self, atoms: Iterable, current, residual, before):
        """Return ``(improvement, projection_score, atom)`` in input order."""
        atoms = list(atoms); results = []
        current_gpu = self.current if hasattr(self, "current") else self.cp.asarray(current)
        residual_gpu = self.residual if hasattr(self, "residual") else self.cp.asarray(residual)
        for start in range(0, len(atoms), self.batch_size):
            chunk = atoms[start:start+self.batch_size]
            basis = _basis_batch(chunk, self.u, self.v, self.cp)
            numerator = self.cp.sum(basis*residual_gpu, axis=(1, 2))
            denominator = self.cp.sum(basis*basis, axis=(1, 2))
            if isinstance(chunk[0], SinusoidComponent):
                amplitudes = self.cp.asarray([a.amplitude for a in chunk])
                projection_gpu = self.cp.mean(
                    (amplitudes[:, None, None]*basis)**2, axis=(1, 2))
            else:
                usable = denominator >= 1e-14
                amplitudes = self.cp.where(usable, numerator/denominator, 0.)
                projection_gpu = amplitudes*amplitudes*denominator/residual.size
            images = current_gpu + amplitudes[:, None, None]*basis
            losses = self._feature_loss(images)
            amplitudes_cpu = self.cp.asnumpy(amplitudes)
            projection = self.cp.asnumpy(projection_gpu)
            losses_cpu = self.cp.asnumpy(losses)
            for atom, amplitude, score, value in zip(
                    chunk, amplitudes_cpu, projection, losses_cpu):
                atom.amplitude = float(amplitude)
                results.append((before-float(value), float(score), atom))
        return results


def group_supported_candidates(candidates):
    """Group without changing the original candidate order."""
    groups = defaultdict(list)
    for index, atom in enumerate(candidates):
        groups[type(atom)].append((index, atom))
    return groups
