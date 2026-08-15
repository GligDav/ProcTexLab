import numpy as np
import pytest

from procedural_texture_kernel import (FitConfig, TextureFitter, compare_spectra,
                                       radial_power_spectrum)


def test_radial_spectrum_retains_absolute_energy():
    y, x = np.indices((64, 64))
    image = np.sin(2 * np.pi * x / 4)
    original = radial_power_spectrum(image, bins=24)
    attenuated = radial_power_spectrum(image * .5, bins=24)
    assert attenuated.non_dc_energy == pytest.approx(original.non_dc_energy * .25)
    assert np.allclose(attenuated.power, original.power * .25)


def test_comparison_detects_high_frequency_deficit():
    y, x = np.indices((64, 64))
    target = np.sin(2 * np.pi * x / 4)
    comparison = compare_spectra(target, target * .5, bins=24)
    assert comparison["high_frequency_ratio"] == pytest.approx(.25)
    assert comparison["frequency_units"] == "fraction_of_axial_nyquist"
    assert [band["name"] for band in comparison["bands"]] == [
        "dc", "very_low", "low", "mid", "high", "very_high"]


def test_identical_and_constant_images_are_numerically_safe():
    image = np.full((9, 13), .4)
    comparison = compare_spectra(image, image)
    assert comparison["high_frequency_ratio"] == 1.0
    assert all(np.isfinite(band["ratio"]) for band in comparison["bands"])
    assert comparison["bands"][0]["ratio"] == pytest.approx(1.0)


def test_spectral_diagnostic_validation():
    with pytest.raises(ValueError, match="finite"):
        radial_power_spectrum(np.array([[np.nan]]))
    with pytest.raises(ValueError, match="matching shapes"):
        compare_spectra(np.zeros((2, 3)), np.zeros((3, 2)))
    with pytest.raises(ValueError, match="bins"):
        radial_power_spectrum(np.zeros((4, 4)), bins=0)


def test_fitter_records_full_resolution_spectral_diagnostics():
    image = np.arange(64, dtype=float).reshape(8, 8)
    result = TextureFitter(FitConfig(max_components=0, fitting_resolution=None)).fit(image)
    diagnostics = result.metadata["spectral_diagnostics"]
    assert len(diagnostics["target"]["frequency"]) == 4
    assert len(diagnostics["target"]["power"]) == 4
    assert np.isfinite(diagnostics["high_frequency_ratio"])
