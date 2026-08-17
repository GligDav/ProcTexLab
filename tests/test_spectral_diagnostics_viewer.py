import numpy as np

from gui.spectral_diagnostics_viewer import BAND_LABELS, _finite_log_range, _format_number


def test_log_plot_range_is_finite_for_zero_and_nonzero_spectra():
    assert _finite_log_range(np.zeros(4), np.zeros(4)) == (-12.0, 0.0)
    lower, upper = _finite_log_range(np.array([1e-8, 1e-4]), np.array([1e-6]))
    assert lower == -8.0
    assert upper == -4.0


def test_spectrum_viewer_formatting_helpers():
    assert _format_number(0.0) == "0"
    assert _format_number(float("inf")) == "n/a"
    assert {"dc", "very_low", "low", "mid", "high", "very_high"} <= set(BAND_LABELS)
