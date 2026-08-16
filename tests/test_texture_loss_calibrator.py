import numpy as np
import pytest

from gui.texture_loss_calibrator import evaluate_images
from procedural_texture_kernel import TextureLossWeights


def test_evaluate_images_reports_loss_for_same_sized_rasters():
    result = evaluate_images(
        np.zeros((8, 12)), np.ones((8, 12)), TextureLossWeights())
    assert result["texture_loss"] > 0
    assert "spectrum_loss" in result
    assert result["mse_loss"] == pytest.approx(1.0)


def test_calibrator_declares_mse_weight():
    from gui.texture_loss_calibrator import WEIGHTS
    assert ("MSE", "mse", "mse_loss", 1.0) in WEIGHTS
    assert ("Local structure", "local_structure", "local_structure_loss", 0.0) in WEIGHTS


def test_evaluate_images_rejects_different_dimensions():
    with pytest.raises(ValueError, match="same dimensions"):
        evaluate_images(np.zeros((8, 12)), np.zeros((12, 8)), TextureLossWeights())
