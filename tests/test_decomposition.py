import numpy as np
import pytest

from procedural_texture_kernel import (FitConfig, LaplacianPyramid, SinusoidComponent,
                                       TextureFitter, create_decomposition)
from procedural_texture_kernel.model import ProceduralTextureModel


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


def test_decomposition_configuration_validation():
    with pytest.raises(ValueError, match="bands"):
        FitConfig(decomposition_bands=0)
    with pytest.raises(ValueError, match="unsupported decomposition"):
        create_decomposition("unknown")
