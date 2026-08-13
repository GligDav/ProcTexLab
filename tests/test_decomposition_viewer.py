import numpy as np

from gui.decomposition_viewer import decompose_image, preview_values


def test_viewer_uses_reconstructable_kernel_decomposition():
    image = np.random.default_rng(9).random((17, 23))
    pyramid, bands = decompose_image(image, bands=5, base_sigma=.8)
    assert len(bands) == 5
    assert np.allclose(pyramid.reconstruct(bands), image, atol=1e-12)


def test_signed_preview_centers_zero_and_scales_symmetrically():
    displayed = preview_values(np.array([[-2.0, 0.0, 1.0]]), signed=True)
    assert np.allclose(displayed, [[0.0, 0.5, 0.75]])
