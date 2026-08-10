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

COMPONENT_TYPES = {c.type_name: c for c in (SinusoidComponent, GaborComponent, GaussianRBFComponent)}

def component_from_dict(data: dict) -> ProceduralComponent:
    """Restore one component from JSON-compatible data."""
    values = dict(data)
    kind = values.pop("type", None)
    try:
        cls = COMPONENT_TYPES[kind]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unsupported component type: {kind!r}") from exc
    return cls(**values)
