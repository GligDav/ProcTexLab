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

COMPONENT_TYPES = {c.type_name: c for c in (
    SinusoidComponent, GaborComponent, GaussianRBFComponent,
    PerlinNoiseComponent, WaveletComponent,
)}

def component_from_dict(data: dict) -> ProceduralComponent:
    """Restore one component from JSON-compatible data."""
    values = dict(data)
    kind = values.pop("type", None)
    try:
        cls = COMPONENT_TYPES[kind]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unsupported component type: {kind!r}") from exc
    return cls(**values)
