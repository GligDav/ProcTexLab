"""Serializable procedural texture model."""
from __future__ import annotations
from dataclasses import dataclass, field
import json
from pathlib import Path
import numpy as np
from .backend import array_module
from .components import ProceduralComponent, component_from_dict
from .coordinates import COORDINATE_SYSTEM, coordinate_grid, coordinate_grid_region

SCHEMA_VERSION = 1

@dataclass
class ProceduralTextureModel:
    """A bias, optional plane, and sparse sum of procedural components."""
    bias: float = 0.0
    trend_u: float = 0.0
    trend_v: float = 0.0
    components: list[ProceduralComponent] = field(default_factory=list)
    def add(self, component: ProceduralComponent) -> None:
        """Append a procedural atom to the model's additive component list."""
        self.components.append(component)
    def evaluate_grid(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Evaluate the trend and all atoms on matching explicit UV grids."""
        if u.shape != v.shape:
            raise ValueError("u and v grids must have the same shape")
        # Establish the global low-frequency surface before accumulating atoms.
        out = self.bias + self.trend_u * (u - 0.5) + self.trend_v * (v - 0.5)
        xp = array_module(u, v)
        # Materialize a writable grid because broadcast views cannot be updated in place.
        out = xp.broadcast_to(out, u.shape).astype(xp.float64, copy=True)
        for component in self.components:
            out += component.evaluate(u, v)
        return out
    def evaluate(self, width: int, height: int) -> np.ndarray:
        """Evaluate to a ``(height, width)`` float array."""
        return self.evaluate_grid(*coordinate_grid(width, height))
    def evaluate_region(
        self, width: int, height: int,
        u_bounds: tuple[float, float] = (0.0, 1.0),
        v_bounds: tuple[float, float] = (0.0, 1.0),
    ) -> np.ndarray:
        """Evaluate an arbitrary UV rectangle for continuation/tiling inspection."""
        return self.evaluate_grid(*coordinate_grid_region(width, height, u_bounds, v_bounds))
    def to_dict(self) -> dict:
        """Return the versioned, JSON-serializable model representation."""
        return {"schema_version": SCHEMA_VERSION, "coordinate_system": COORDINATE_SYSTEM,
                "bias": self.bias, "trend_u": self.trend_u, "trend_v": self.trend_v,
                "components": [c.to_dict() for c in self.components]}
    @classmethod
    def from_dict(cls, data: dict) -> "ProceduralTextureModel":
        """Validate and deserialize a model mapping produced by :meth:`to_dict`."""
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema version: {data.get('schema_version')!r}")
        if data.get("coordinate_system") != COORDINATE_SYSTEM:
            raise ValueError("unsupported coordinate system")
        return cls(float(data.get("bias", 0)), float(data.get("trend_u", 0)),
                   float(data.get("trend_v", 0)),
                   [component_from_dict(x) for x in data.get("components", [])])
    def save_json(self, path: str | Path) -> None:
        """Serialize the model as indented JSON at ``path``."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
    @classmethod
    def load_json(cls, path: str | Path) -> "ProceduralTextureModel":
        """Load and validate a model from a UTF-8 JSON file."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
