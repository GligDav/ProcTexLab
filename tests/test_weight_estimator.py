import numpy as np
import pytest

from procedural_texture_kernel import WeightEstimator


def patterns(size=64):
    y, x = np.indices((size, size))
    rng = np.random.default_rng(42)
    return {
        "constant": np.full((size, size), 3.5),
        "noise": rng.normal(size=(size, size)),
        "sinusoid": np.sin(2 * np.pi * x / 8),
        "stripes": np.sign(np.sin(2 * np.pi * x / 8)),
        "impulses": np.where(rng.random((size, size)) < .015, 20.0, 0.0),
        "smooth": np.cos(np.pi * x / size) + np.cos(np.pi * y / size),
    }


@pytest.mark.parametrize("name", patterns())
def test_features_and_weights_are_numerically_valid(name):
    result = WeightEstimator().analyze(patterns()[name])
    normalized = [result.features.spectral_entropy, result.features.spectral_anisotropy,
                  result.features.autocorrelation_strength, result.features.gradient_coherence,
                  result.features.kurtosis]
    weights = list(result.weights.to_dict().values())
    assert np.all(np.isfinite(normalized + weights))
    assert np.all((np.asarray(normalized) >= 0) & (np.asarray(normalized) <= 1))
    assert np.all(np.asarray(weights) >= 0)
    assert sum(weights) == pytest.approx(1.0, abs=1e-15)


def test_noise_has_higher_entropy_than_periodic_pattern():
    p = patterns()
    estimator = WeightEstimator()
    assert (estimator.analyze(p["noise"]).features.spectral_entropy >
            estimator.analyze(p["sinusoid"]).features.spectral_entropy)


def test_directional_stripes_are_more_anisotropic_and_coherent_than_noise():
    p = patterns(); estimator = WeightEstimator()
    stripes = estimator.analyze(p["stripes"]).features
    noise = estimator.analyze(p["noise"]).features
    assert stripes.spectral_anisotropy > noise.spectral_anisotropy
    assert stripes.gradient_coherence > noise.gradient_coherence


def test_periodic_pattern_has_off_center_autocorrelation():
    p = patterns(); estimator = WeightEstimator()
    assert (estimator.analyze(p["sinusoid"]).features.autocorrelation_strength >
            estimator.analyze(p["noise"]).features.autocorrelation_strength)


def test_sparse_impulses_have_stronger_kurtosis_descriptor_than_noise():
    p = patterns(); estimator = WeightEstimator()
    assert (estimator.analyze(p["impulses"]).features.kurtosis >
            estimator.analyze(p["noise"]).features.kurtosis)


@pytest.mark.parametrize("image", [np.array([[2.0]]), np.zeros((1, 7)), np.full((4, 3), 1e-300),
                                   np.array([[1e300, -1e300], [-1e300, 1e300]])])
def test_tiny_and_near_zero_bands_are_safe(image):
    result = WeightEstimator().analyze(image)
    assert sum(result.weights.to_dict().values()) == pytest.approx(1.0)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_non_finite_input_is_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        WeightEstimator().analyze(np.array([[0.0, value]]))
