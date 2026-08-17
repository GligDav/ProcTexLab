"""Procedural atom definitions, independent of fitting and user interfaces."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import ClassVar
import numpy as np

@dataclass
class ProceduralComponent:
    """Base class for an explicitly parameterized scalar atom."""
    amplitude: float = 1.0
    type_name: ClassVar[str] = "component"
    def basis(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    def evaluate(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return self.amplitude * self.basis(u, v)
    def to_dict(self) -> dict:
        return {"type": self.type_name, **asdict(self)}

@dataclass
class SinusoidComponent(ProceduralComponent):
    """Global cosine; frequencies are cycles per normalized U/V interval."""
    frequency_u: float = 1.0
    frequency_v: float = 0.0
    phase: float = 0.0
    type_name: ClassVar[str] = "sinusoid"
    def basis(self, u, v):
        return np.cos(2.0 * np.pi * (self.frequency_u * u + self.frequency_v * v) + self.phase)

@dataclass
class GaborComponent(ProceduralComponent):
    """Gaussian-windowed oriented cosine atom."""
    center_u: float = 0.5
    center_v: float = 0.5
    sigma_u: float = 0.15
    sigma_v: float = 0.15
    frequency: float = 4.0
    orientation: float = 0.0
    phase: float = 0.0
    type_name: ClassVar[str] = "gabor"
    def basis(self, u, v):
        du, dv = u - self.center_u, v - self.center_v
        c, s = np.cos(self.orientation), np.sin(self.orientation)
        x, y = c * du + s * dv, -s * du + c * dv
        envelope = np.exp(-0.5 * ((x / self.sigma_u) ** 2 + (y / self.sigma_v) ** 2))
        return envelope * np.cos(2.0 * np.pi * self.frequency * x + self.phase)

@dataclass
class GaussianRBFComponent(ProceduralComponent):
    """Localized Gaussian radial basis atom."""
    center_u: float = 0.5
    center_v: float = 0.5
    sigma: float = 0.1
    type_name: ClassVar[str] = "gaussian_rbf"
    def basis(self, u, v):
        r2 = (u - self.center_u) ** 2 + (v - self.center_v) ** 2
        return np.exp(-0.5 * r2 / self.sigma**2)

@dataclass
class PerlinNoiseComponent(ProceduralComponent):
    """Seeded fractal 2D Perlin gradient noise."""
    frequency: float = 4.0
    octaves: int = 1
    persistence: float = 0.5
    lacunarity: float = 2.0
    offset_u: float = 0.0
    offset_v: float = 0.0
    seed: int = 0
    type_name: ClassVar[str] = "perlin_noise"

    @staticmethod
    def _noise(x, y, permutation):
        xi = np.floor(x).astype(np.int64) & 255
        yi = np.floor(y).astype(np.int64) & 255
        xf, yf = x - np.floor(x), y - np.floor(y)
        fade = lambda t: t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
        sx, sy = fade(xf), fade(yf)
        p = permutation
        aa, ab = p[p[xi] + yi], p[p[xi] + yi + 1]
        ba, bb = p[p[xi + 1] + yi], p[p[xi + 1] + yi + 1]
        def gradient(h, dx, dy):
            angle = (h & 7) * (np.pi / 4.0)
            return np.cos(angle) * dx + np.sin(angle) * dy
        lerp = lambda a, b, t: a + t * (b - a)
        lower = lerp(gradient(aa, xf, yf), gradient(ba, xf - 1, yf), sx)
        upper = lerp(gradient(ab, xf, yf - 1), gradient(bb, xf - 1, yf - 1), sx)
        return np.sqrt(2.0) * lerp(lower, upper, sy)

    def basis(self, u, v):
        if self.octaves < 1:
            raise ValueError("Perlin octaves must be at least one")
        rng = np.random.default_rng(self.seed)
        base = rng.permutation(256)
        permutation = np.concatenate((base, base))
        result = np.zeros(np.broadcast_shapes(np.shape(u), np.shape(v)), dtype=np.float64)
        weight = 1.0
        frequency = self.frequency
        for _ in range(self.octaves):
            result += weight * self._noise(
                (u + self.offset_u) * frequency,
                (v + self.offset_v) * frequency,
                permutation,
            )
            weight *= self.persistence
            frequency *= self.lacunarity
        total_weight = sum(self.persistence ** i for i in range(self.octaves))
        return result / total_weight

@dataclass
class WaveletComponent(ProceduralComponent):
    """Localized anisotropic Mexican-hat (Ricker) wavelet atom."""
    center_u: float = 0.5
    center_v: float = 0.5
    scale_u: float = 0.12
    scale_v: float = 0.12
    orientation: float = 0.0
    type_name: ClassVar[str] = "wavelet"
    def basis(self, u, v):
        du, dv = u - self.center_u, v - self.center_v
        c, s = np.cos(self.orientation), np.sin(self.orientation)
        x, y = c * du + s * dv, -s * du + c * dv
        radius2 = (x / self.scale_u) ** 2 + (y / self.scale_v) ** 2
        return (1.0 - radius2) * np.exp(-0.5 * radius2)

def _perlin_basis(u, v, frequency, octaves, persistence, lacunarity,
                  offset_u, offset_v, seed):
    """Shared deterministic octave-noise implementation."""
    return PerlinNoiseComponent(1.0, frequency, octaves, persistence, lacunarity,
                                offset_u, offset_v, seed).basis(u, v)

@dataclass
class ThresholdedNoiseComponent(ProceduralComponent):
    """Smoothly thresholded, rotated fBm field for coherent binary regions."""
    frequency: float = 2.0
    octaves: int = 4
    persistence: float = 0.5
    lacunarity: float = 2.0
    offset_u: float = 0.0
    offset_v: float = 0.0
    rotation: float = 0.0
    threshold: float = 0.0
    edge_width: float = 0.08
    seed: int = 0
    type_name: ClassVar[str] = "thresholded_noise"

    def basis(self, u, v):
        # Rotate around the source-domain center while retaining procedural
        # continuation outside [0, 1).  Offsets remain in rotated UV space.
        du, dv = u - .5, v - .5
        c, s = np.cos(self.rotation), np.sin(self.rotation)
        ru = c * du + s * dv + .5
        rv = -s * du + c * dv + .5
        noise = _perlin_basis(ru, rv, self.frequency, self.octaves,
                              self.persistence, self.lacunarity,
                              self.offset_u, self.offset_v, self.seed)
        width = max(float(self.edge_width), 1e-12)
        t = np.clip((noise - (self.threshold - width)) / (2.0 * width), 0.0, 1.0)
        smooth = t * t * (3.0 - 2.0 * t)
        return 2.0 * smooth - 1.0


@dataclass
class MaskedNoiseComponent(ProceduralComponent):
    """Detail noise restricted to one side of a coherent thresholded-noise mask."""
    mask_frequency: float = 2.0
    mask_octaves: int = 4
    mask_offset_u: float = 0.0
    mask_offset_v: float = 0.0
    mask_rotation: float = 0.0
    mask_threshold: float = 0.0
    mask_edge_width: float = 0.08
    mask_seed: int = 0
    detail_frequency: float = 8.0
    detail_octaves: int = 3
    detail_offset_u: float = 0.0
    detail_offset_v: float = 0.0
    detail_seed: int = 1
    invert_mask: bool = False
    type_name: ClassVar[str] = "masked_noise"

    def basis(self, u, v):
        mask = ThresholdedNoiseComponent(
            frequency=self.mask_frequency, octaves=self.mask_octaves,
            offset_u=self.mask_offset_u, offset_v=self.mask_offset_v,
            rotation=self.mask_rotation, threshold=self.mask_threshold,
            edge_width=self.mask_edge_width, seed=self.mask_seed).basis(u, v)
        mask = .5 * (mask + 1.0)
        if self.invert_mask:
            mask = 1.0 - mask
        detail = _perlin_basis(
            u, v, self.detail_frequency, self.detail_octaves, .5, 2.0,
            self.detail_offset_u, self.detail_offset_v, self.detail_seed)
        return mask * detail

@dataclass
class VoronoiNoiseComponent(ProceduralComponent):
    """Seeded cellular (Worley/Voronoi) nearest-feature noise."""
    frequency: float = 5.0
    jitter: float = 1.0
    offset_u: float = 0.0
    offset_v: float = 0.0
    seed: int = 0
    type_name: ClassVar[str] = "voronoi_noise"
    def basis(self, u, v):
        x = (u + self.offset_u) * self.frequency
        y = (v + self.offset_v) * self.frequency
        ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
        best = np.full(np.broadcast_shapes(np.shape(u), np.shape(v)), np.inf)
        def rnd(a, b, salt):
            n = np.sin(a * 127.1 + b * 311.7 + (self.seed + salt) * 74.7) * 43758.5453
            return n - np.floor(n)
        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                cx, cy = ix + ox, iy + oy
                px = cx + .5 + (rnd(cx, cy, 0) - .5) * self.jitter
                py = cy + .5 + (rnd(cx, cy, 1) - .5) * self.jitter
                best = np.minimum(best, np.hypot(x - px, y - py))
        return np.clip(1.0 - np.sqrt(2.0) * best, -1.0, 1.0)

@dataclass
class FractalBrownianMotionComponent(ProceduralComponent):
    """Normalized fractal Brownian motion made from Perlin octaves."""
    frequency: float = 2.0
    octaves: int = 5
    persistence: float = 0.5
    lacunarity: float = 2.0
    offset_u: float = 0.0
    offset_v: float = 0.0
    seed: int = 0
    type_name: ClassVar[str] = "fbm"
    def basis(self, u, v):
        return _perlin_basis(u, v, self.frequency, self.octaves, self.persistence,
                             self.lacunarity, self.offset_u, self.offset_v, self.seed)

@dataclass
class RidgedMultifractalComponent(FractalBrownianMotionComponent):
    """Sharp ridges obtained by folding each noise octave."""
    ridge_offset: float = 1.0
    ridge_power: float = 2.0
    rotation: float = 0.0
    anisotropy: float = 1.0
    type_name: ClassVar[str] = "ridged_multifractal"
    def basis(self, u, v):
        if self.octaves < 1:
            raise ValueError("Ridged multifractal octaves must be at least one")
        rng = np.random.default_rng(self.seed)
        base = rng.permutation(256)
        permutation = np.concatenate((base, base))
        du, dv = u - .5, v - .5
        c, s = np.cos(self.rotation), np.sin(self.rotation)
        ru = c * du + s * dv + .5 + self.offset_u
        rv = (-s * du + c * dv) * max(float(self.anisotropy), 1e-6) + .5 + self.offset_v
        result = np.zeros(np.broadcast_shapes(np.shape(u), np.shape(v)), dtype=np.float64)
        weight, frequency, total_weight = 1.0, self.frequency, 0.0
        ridge_power = max(float(self.ridge_power), 1e-6)
        for _ in range(self.octaves):
            noise = PerlinNoiseComponent._noise(
                ru * frequency, rv * frequency, permutation)
            ridge = np.clip(self.ridge_offset - np.abs(noise), 0.0, 1.0) ** ridge_power
            result += weight * ridge
            total_weight += weight
            weight *= self.persistence
            frequency *= self.lacunarity
        return 2.0 * result / max(total_weight, 1e-12) - 1.0

@dataclass
class TurbulenceNoiseComponent(FractalBrownianMotionComponent):
    """Absolute-value (folded) fractal noise."""
    type_name: ClassVar[str] = "turbulence_noise"
    def basis(self, u, v):
        return 2.0 * np.abs(super().basis(u, v)) - 1.0

@dataclass
class DomainWarpedNoiseComponent(FractalBrownianMotionComponent):
    """fBm sampled through a second seeded vector-valued noise field."""
    warp_amplitude: float = 0.15
    warp_frequency: float = 2.0
    type_name: ClassVar[str] = "domain_warped_noise"
    def basis(self, u, v):
        wu = _perlin_basis(u, v, self.warp_frequency, 3, .5, 2., 0., 0., self.seed + 101)
        wv = _perlin_basis(u, v, self.warp_frequency, 3, .5, 2., 0., 0., self.seed + 211)
        return _perlin_basis(u + self.warp_amplitude * wu, v + self.warp_amplitude * wv,
                             self.frequency, self.octaves, self.persistence,
                             self.lacunarity, self.offset_u, self.offset_v, self.seed)

def _rotated(u, v, center_u, center_v, orientation):
    du, dv = u - center_u, v - center_v
    c, s = np.cos(orientation), np.sin(orientation)
    return c * du + s * dv, -s * du + c * dv

@dataclass
class AnisotropicGaussianComponent(ProceduralComponent):
    """Rotated elliptical Gaussian blob."""
    center_u: float = .5
    center_v: float = .5
    sigma_u: float = .15
    sigma_v: float = .08
    orientation: float = 0.0
    type_name: ClassVar[str] = "anisotropic_gaussian"
    def basis(self, u, v):
        x, y = _rotated(u, v, self.center_u, self.center_v, self.orientation)
        return np.exp(-.5 * ((x / self.sigma_u) ** 2 + (y / self.sigma_v) ** 2))

@dataclass
class LineComponent(ProceduralComponent):
    """Finite or infinite soft-edged ridge/bar."""
    center_u: float = .5
    center_v: float = .5
    width: float = .04
    length: float = 1.0
    orientation: float = 0.0
    softness: float = .01
    type_name: ClassVar[str] = "line"
    def basis(self, u, v):
        x, y = _rotated(u, v, self.center_u, self.center_v, self.orientation)
        d = np.maximum(np.abs(y) - self.width / 2, np.abs(x) - self.length / 2)
        return 1.0 / (1.0 + np.exp(np.clip(d / max(self.softness, 1e-12), -60, 60)))

@dataclass
class StepEdgeComponent(ProceduralComponent):
    """Oriented hard step or smooth sigmoid edge."""
    center_u: float = .5
    center_v: float = .5
    orientation: float = 0.0
    softness: float = .02
    type_name: ClassVar[str] = "step_edge"
    def basis(self, u, v):
        x, _ = _rotated(u, v, self.center_u, self.center_v, self.orientation)
        if self.softness <= 0:
            return np.where(x >= 0, 1.0, -1.0)
        return np.tanh(x / self.softness)

@dataclass
class DifferenceOfGaussiansComponent(ProceduralComponent):
    """Difference-of-Gaussians or analytic Laplacian-of-Gaussian atom."""
    center_u: float = .5
    center_v: float = .5
    sigma: float = .1
    ratio: float = 1.6
    mode: str = "dog"
    type_name: ClassVar[str] = "dog_log"
    def basis(self, u, v):
        r2 = (u - self.center_u) ** 2 + (v - self.center_v) ** 2
        q = r2 / self.sigma**2
        if self.mode == "log":
            return (1.0 - .5 * q) * np.exp(-.5 * q)
        if self.mode != "dog":
            raise ValueError("mode must be 'dog' or 'log'")
        outer = max(self.ratio, 1.000001) * self.sigma
        return np.exp(-.5 * q) - np.exp(-.5 * r2 / outer**2) / self.ratio**2

@dataclass
class PolynomialTrendComponent(ProceduralComponent):
    """Global quadratic/low-frequency polynomial trend."""
    linear_u: float = 0.0
    linear_v: float = 0.0
    quadratic_u: float = 1.0
    cross_uv: float = 0.0
    quadratic_v: float = 1.0
    type_name: ClassVar[str] = "polynomial_trend"
    def basis(self, u, v):
        x, y = u - .5, v - .5
        return (self.linear_u*x + self.linear_v*y + self.quadratic_u*x*x +
                self.cross_uv*x*y + self.quadratic_v*y*y)

@dataclass
class RadialWaveComponent(ProceduralComponent):
    """Concentric cosine wave."""
    center_u: float = .5
    center_v: float = .5
    frequency: float = 8.0
    phase: float = 0.0
    decay: float = 0.0
    type_name: ClassVar[str] = "radial_wave"
    def basis(self, u, v):
        r = np.hypot(u - self.center_u, v - self.center_v)
        return np.cos(2*np.pi*self.frequency*r + self.phase) * np.exp(-self.decay*r)

@dataclass
class SpiralWaveComponent(RadialWaveComponent):
    """Curved wave with radial and angular phase progression."""
    arms: float = 2.0
    type_name: ClassVar[str] = "spiral_wave"
    def basis(self, u, v):
        du, dv = u - self.center_u, v - self.center_v
        r, theta = np.hypot(du, dv), np.arctan2(dv, du)
        return np.cos(2*np.pi*self.frequency*r + self.arms*theta + self.phase) * np.exp(-self.decay*r)

@dataclass
class SparseImpulseComponent(ProceduralComponent):
    """Deterministic sparse field of Gaussian spots."""
    density: float = 12.0
    radius: float = .025
    seed: int = 0
    signed: bool = False
    type_name: ClassVar[str] = "sparse_impulse"
    def basis(self, u, v):
        count = max(0, int(round(self.density)))
        rng = np.random.default_rng(self.seed)
        out = np.zeros(np.broadcast_shapes(np.shape(u), np.shape(v)))
        for _ in range(count):
            cu, cv = rng.random(2); sign = rng.choice((-1., 1.)) if self.signed else 1.
            out += sign * np.exp(-.5 * ((u-cu)**2 + (v-cv)**2) / max(self.radius, 1e-12)**2)
        return np.clip(out, -1., 1.)

@dataclass
class BinaryPrimitiveComponent(ProceduralComponent):
    """Hard binary disk, box, ring, or checker primitive."""
    center_u: float = .5
    center_v: float = .5
    size_u: float = .2
    size_v: float = .2
    orientation: float = 0.0
    shape: str = "disk"
    thickness: float = .03
    type_name: ClassVar[str] = "binary_primitive"
    def basis(self, u, v):
        x, y = _rotated(u, v, self.center_u, self.center_v, self.orientation)
        if self.shape == "disk": mask = (x/self.size_u)**2 + (y/self.size_v)**2 <= 1
        elif self.shape == "box": mask = (np.abs(x) <= self.size_u) & (np.abs(y) <= self.size_v)
        elif self.shape == "ring": mask = np.abs(np.sqrt((x/self.size_u)**2 + (y/self.size_v)**2)-1) <= self.thickness/max(self.size_u, self.size_v)
        elif self.shape == "checker": mask = (np.floor(x/self.size_u) + np.floor(y/self.size_v)).astype(int) % 2 == 0
        else: raise ValueError("shape must be 'disk', 'box', 'ring', or 'checker'")
        return mask.astype(float)

@dataclass
class SimpleConstantComponent(ProceduralComponent):
    """Simple, monochromic constant component."""
    value:float = 0.0
    type_name: ClassVar[str] = "simple_constant"
    def basis(self, u, v):
        return np.ones(np.broadcast_shapes(np.shape(u), np.shape(v))) * self.value

COMPONENT_TYPES = {c.type_name: c for c in (
    SinusoidComponent, GaborComponent, GaussianRBFComponent,
    PerlinNoiseComponent, ThresholdedNoiseComponent, MaskedNoiseComponent, WaveletComponent,
    VoronoiNoiseComponent, FractalBrownianMotionComponent,
    RidgedMultifractalComponent, TurbulenceNoiseComponent, DomainWarpedNoiseComponent,
    AnisotropicGaussianComponent, LineComponent, StepEdgeComponent,
    DifferenceOfGaussiansComponent, PolynomialTrendComponent, RadialWaveComponent,
    SpiralWaveComponent, SparseImpulseComponent, BinaryPrimitiveComponent, SimpleConstantComponent,
)}

def component_from_dict(data: dict) -> ProceduralComponent:
    """Restore one component from JSON-compatible data."""
    values = dict(data)
    kind = values.pop("type", None)
    if kind == "shader_graph":
        from .shader_graph import ShaderGraphComponent
        return ShaderGraphComponent.from_dict(values)
    try:
        cls = COMPONENT_TYPES[kind]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unsupported component type: {kind!r}") from exc
    return cls(**values)
