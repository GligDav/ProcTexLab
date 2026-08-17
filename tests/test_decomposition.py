import numpy as np
import pytest

from procedural_texture_kernel import (FitConfig, LaplacianPyramid, SinusoidComponent,
                                       TextureFitter, create_decomposition)
from procedural_texture_kernel.model import ProceduralTextureModel
from procedural_texture_kernel.fitting import _families_for_band


@pytest.mark.parametrize("shape,bands", [((31, 47), 1), ((32, 32), 5), ((9, 15), 7)])
def test_laplacian_pyramid_reconstructs_input(shape, bands):
    image = np.random.default_rng(12).normal(size=shape)
    pyramid = LaplacianPyramid(bands=bands)
    decomposed = pyramid.decompose(image)
    assert len(decomposed) == bands
    assert all(band.shape == shape for band in decomposed)
    assert np.allclose(pyramid.reconstruct(decomposed), image, rtol=1e-12, atol=1e-12)


def test_laplacian_scales_are_octave_spaced():
    assert LaplacianPyramid(bands=5, base_sigma=.75).sigmas == (.75, 1.5, 3.0, 6.0)


def test_band_aware_candidate_roles_follow_pyramid_order():
    config = FitConfig()
    high = _families_for_band(config, 0, 5)
    middle = _families_for_band(config, 2, 5)
    low = _families_for_band(config, 4, 5)
    assert "wavelet" in high and "thresholded_noise" not in high
    assert "wavelet" in middle and "thresholded_noise" in middle
    assert "thresholded_noise" in low and "wavelet" not in low


def test_band_roles_preserve_single_explicit_family_and_can_be_disabled():
    selected = ("sinusoid",)
    config = FitConfig(component_families=selected)
    assert _families_for_band(config, 4, 5) == selected
    disabled = FitConfig(component_families=("sinusoid", "thresholded_noise"),
                         band_aware_candidates=False)
    assert _families_for_band(disabled, 0, 5) == disabled.component_families


def test_fitter_optimizes_each_target_band_independently():
    image = ProceduralTextureModel(
        .5, components=[SinusoidComponent(.2, 3, 1, .2)]).evaluate(24, 20)
    result = TextureFitter(FitConfig(
        decomposition_bands=3, max_components=1, max_iterations=2,
        fitting_resolution=None, component_families=("sinusoid",),
        min_improvement=0)).fit(image)
    assert len(result.metadata["bands"]) == 3
    assert all(entry["band"] == index + 1
               for index, entry in enumerate(result.metadata["bands"]))
    assert len(result.model.components) == sum(
        entry["components"] for entry in result.metadata["bands"])


def test_fitter_estimates_and_records_weights_for_each_band():
    y, x = np.indices((32, 40))
    image = .5 + .2 * np.sin(2 * np.pi * x / 8) + .05 * np.random.default_rng(4).normal(size=x.shape)
    result = TextureFitter(FitConfig(decomposition_bands=3, max_components=0,
                                     fitting_resolution=None)).fit(image)
    bands = result.metadata["bands"]
    assert result.metadata["objective"]["weight_mode"] == "adaptive_per_band"
    assert all("features" in band and "weights" in band for band in bands)
    for band in bands:
        statistical = [band["weights"][name] for name in
                       ("spectrum", "histogram", "autocorrelation", "gradient")]
        assert np.all(np.isfinite(statistical))
        assert sum(statistical) == pytest.approx(1.0)
        assert band["weights"]["mse"] == 1.0
    assert bands[0]["weights"] != bands[-1]["weights"]


def test_manual_band_weights_remain_available():
    config = FitConfig(decomposition_bands=2, max_components=0,
                       adaptive_texture_weights=False, spectrum_weight=2,
                       histogram_weight=3, autocorrelation_weight=4,
                       gradient_weight=5, mse_weight=6)
    result = TextureFitter(config).fit(np.arange(64, dtype=float).reshape(8, 8))
    assert result.metadata["objective"]["weight_mode"] == "manual"
    assert all(band["weights"] == {"spectrum": 2, "histogram": 3,
                                   "autocorrelation": 4, "gradient": 5, "mse": 6,
                                   "local_structure": 0, "local_contrast": 0}
               for band in result.metadata["bands"])
    assert all("features" not in band for band in result.metadata["bands"])


def test_decomposition_configuration_validation():
    with pytest.raises(ValueError, match="bands"):
        FitConfig(decomposition_bands=0)
    with pytest.raises(ValueError, match="unsupported decomposition"):
        create_decomposition("unknown")


def test_high_frequency_refinement_adds_detail_when_base_fit_has_no_atoms():
    y, x = np.indices((48, 48))
    image = .5 + .2 * np.sin(2 * np.pi * x / 4)
    result = TextureFitter(FitConfig(
        decomposition_bands=1, max_components=0, fitting_resolution=None,
        component_families=("sinusoid",), max_frequency=20,
        detail_refinement=True, detail_max_components=1,
        detail_min_frequency=8, detail_hf_ratio_threshold=.9,
        max_iterations=20, min_improvement=0)).fit(image)
    detail = result.metadata["detail_refinement"]
    assert detail["attempted"]
    assert detail["accepted"]
    assert detail["components"] == 1
    assert detail["after_hf_absolute_error"] < detail["before_hf_absolute_error"]
    assert detail["after_mse"] <= detail["before_mse"]


def test_high_frequency_refinement_skips_when_threshold_is_met():
    image = np.full((16, 16), .4)
    result = TextureFitter(FitConfig(
        max_components=0, fitting_resolution=None,
        detail_refinement=True, detail_min_frequency=6)).fit(image)
    detail = result.metadata["detail_refinement"]
    assert not detail["attempted"]
    assert detail["reason"] == "high_frequency_ratio_meets_threshold"


def test_joint_amplitude_refit_is_recorded_and_never_worsens_band_objective():
    y, x = np.indices((32, 40))
    image = (.5 + .18 * np.sin(2 * np.pi * x / 8)
             + .1 * np.sin(2 * np.pi * (x + y) / 10))
    result = TextureFitter(FitConfig(
        decomposition_bands=1, max_components=2, fitting_resolution=None,
        component_families=("sinusoid",), min_improvement=0,
        joint_amplitude_refit=True, amplitude_refit_interval=1,
        max_iterations=10)).fit(image)
    iterations = result.metadata["bands"][0]["iterations"]
    assert iterations
    assert all(item["amplitude_refit"]["attempted"] for item in iterations)
    assert all(item["amplitude_refit"]["after"]
               <= item["amplitude_refit"]["before"] + 1e-15
               for item in iterations)
