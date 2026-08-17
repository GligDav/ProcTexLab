"""Absolute radial power-spectrum diagnostics for reconstruction quality."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


_BAND_LIMITS = (
    ("very_low", 0.0, 0.125),
    ("low", 0.125, 0.25),
    ("mid", 0.25, 0.5),
    ("high", 0.5, 0.75),
    ("very_high", 0.75, np.sqrt(2.0) + 1e-12),
)


@dataclass(frozen=True)
class RadialPowerSpectrum:
    """Radially averaged PSD with frequency expressed as a Nyquist fraction."""

    frequency: np.ndarray
    power: np.ndarray
    sample_count: np.ndarray
    dc_energy: float
    non_dc_energy: float

    def to_dict(self) -> dict:
        return {"frequency": self.frequency.tolist(), "power": self.power.tolist(),
                "sample_count": self.sample_count.tolist(),
                "dc_energy": self.dc_energy, "non_dc_energy": self.non_dc_energy}


def _validate_image(image: np.ndarray, name: str = "image") -> np.ndarray:
    values = np.asarray(image, dtype=np.float64)
    if values.ndim != 2 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite, non-empty 2D image")
    return values


def _hann(length: int) -> np.ndarray:
    return np.ones(1) if length == 1 else np.hanning(length)


def _power_spectrum(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return window-corrected 2D PSD, Nyquist-relative radius and DC energy."""
    height, width = image.shape
    centered = image - np.mean(image)
    window = np.outer(_hann(height), _hann(width))
    window_power = max(float(np.mean(window * window)), np.finfo(float).tiny)
    transformed = np.fft.fft2(centered * window)
    power = np.abs(transformed) ** 2 / (image.size ** 2 * window_power)
    fy = np.fft.fftfreq(height)
    fx = np.fft.fftfreq(width)
    radius = np.hypot(fy[:, None], fx[None, :]) / 0.5
    return power, radius, float(np.mean(image) ** 2)


def radial_power_spectrum(image: np.ndarray, bins: int | None = None) -> RadialPowerSpectrum:
    """Calculate an absolute radial PSD without normalizing away image energy."""
    values = _validate_image(image)
    if bins is None:
        bins = max(4, min(values.shape) // 2)
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer")
    power, radius, dc_energy = _power_spectrum(values)
    edges = np.linspace(0.0, np.sqrt(2.0), bins + 1)
    indices = np.minimum(np.searchsorted(edges, radius.ravel(), side="right") - 1,
                         bins - 1)
    indices = np.maximum(indices, 0)
    counts = np.bincount(indices, minlength=bins)
    sums = np.bincount(indices, weights=power.ravel(), minlength=bins)
    radial = np.divide(sums, counts, out=np.zeros(bins), where=counts > 0)
    return RadialPowerSpectrum((edges[:-1] + edges[1:]) * 0.5, radial, counts,
                               dc_energy, float(np.sum(power)))


def _safe_ratio(target: float, result: float) -> float:
    scale = max(abs(target), abs(result), 1.0)
    epsilon = np.finfo(float).eps * scale
    if abs(target) <= epsilon and abs(result) <= epsilon:
        return 1.0
    return float(result / max(target, epsilon))


def compare_spectra(target: np.ndarray, result: np.ndarray,
                    bins: int | None = None) -> dict:
    """Compare absolute target/result energy in radial frequency bands.

    Frequencies are fractions of the axial Nyquist frequency. Diagonal FFT
    samples can therefore reach ``sqrt(2)``. DC is reported separately from
    the mean-removed texture spectrum.
    """
    reference = _validate_image(target, "target")
    candidate = _validate_image(result, "result")
    if reference.shape != candidate.shape:
        raise ValueError("target and result must have matching shapes")
    target_power, radius, target_dc = _power_spectrum(reference)
    result_power, _, result_dc = _power_spectrum(candidate)
    target_total = float(np.sum(target_power))
    result_total = float(np.sum(result_power))
    target_all = target_dc + target_total
    result_all = result_dc + result_total
    bands = [{"name": "dc", "frequency_min": 0.0, "frequency_max": 0.0,
              "target_energy": target_dc, "result_energy": result_dc,
              "target_fraction": target_dc / max(target_all, np.finfo(float).tiny),
              "result_fraction": result_dc / max(result_all, np.finfo(float).tiny),
              "ratio": _safe_ratio(target_dc, result_dc)}]
    for name, lower, upper in _BAND_LIMITS:
        mask = (radius >= lower) & (radius < upper)
        target_energy = float(np.sum(target_power[mask]))
        result_energy = float(np.sum(result_power[mask]))
        bands.append({"name": name, "frequency_min": lower, "frequency_max": upper,
                      "target_energy": target_energy, "result_energy": result_energy,
                      "target_fraction": target_energy / max(target_all, np.finfo(float).tiny),
                      "result_fraction": result_energy / max(result_all, np.finfo(float).tiny),
                      "ratio": _safe_ratio(target_energy, result_energy)})
    high_target = sum(x["target_energy"] for x in bands if x["name"] in ("high", "very_high"))
    high_result = sum(x["result_energy"] for x in bands if x["name"] in ("high", "very_high"))
    return {"frequency_units": "fraction_of_axial_nyquist",
            "target": radial_power_spectrum(reference, bins).to_dict(),
            "result": radial_power_spectrum(candidate, bins).to_dict(),
            "bands": bands, "high_frequency_ratio": _safe_ratio(high_target, high_result)}
