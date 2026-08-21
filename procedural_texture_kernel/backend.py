"""Optional numerical backends shared by model and fitting code."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter as numpy_gaussian_filter
from scipy.ndimage import zoom as numpy_zoom


def load_cupy():
    """Import CuPy lazily and verify that a CUDA device is usable."""
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
        probe = cp.asarray((1.0,), dtype=cp.float64)
        cp.sum(probe).get()
    except Exception as exc:
        if isinstance(exc, RuntimeError) and str(exc).startswith("CuPy is installed"):
            raise
        raise RuntimeError("CuPy could not initialize a CUDA device") from exc
    return cp, gaussian_filter


def array_module(*values):
    """Return CuPy for CUDA arrays and NumPy otherwise, without eager imports."""
    if any(hasattr(value, "__cuda_array_interface__") for value in values):
        import cupy as cp
        return cp
    return np


@dataclass(frozen=True)
class NumericBackend:
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
    """Initialize the explicitly requested numerical backend."""
    if name == "numpy":
        return NumericBackend("numpy", np, numpy_gaussian_filter, numpy_zoom)
    if name != "cupy":
        raise ValueError("compute backend must be 'numpy' or 'cupy'")
    cp, gaussian_filter = load_cupy()
    try:
        from cupyx.scipy.ndimage import zoom
    except ImportError as exc:
        raise RuntimeError(
            "the installed CuPy package does not provide cupyx.scipy.ndimage") from exc
    return NumericBackend("cupy", cp, gaussian_filter, zoom)
