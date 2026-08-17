import numpy as np
import pytest

from procedural_texture_kernel import compare_measurements


def test_identical_images_have_unit_measurement_ratios():
    y, x = np.indices((32, 32))
    image = np.sin(2 * np.pi * x / 8) + .25 * np.cos(2 * np.pi * y / 5)
    diagnostics = compare_measurements(image, image)
    assert all(row["ratio"] == pytest.approx(1.0) for row in diagnostics["summary"])
    assert all(row["mean_ratio"] == pytest.approx(1.0)
               and row["p95_ratio"] == pytest.approx(1.0)
               for row in diagnostics["local_contrast"])
    assert all(row["ratio"] == pytest.approx(1.0)
               for row in diagnostics["oriented_spectrum"])


def test_attenuation_is_visible_in_absolute_measurements():
    y, x = np.indices((32, 32))
    target = np.sin(2 * np.pi * x / 4)
    diagnostics = compare_measurements(target, target * .5)
    summary = {row["name"]: row for row in diagnostics["summary"]}
    assert summary["Standard deviation"]["ratio"] == pytest.approx(.5)
    assert summary["Gradient RMS"]["ratio"] == pytest.approx(.5)
    active = max(diagnostics["oriented_spectrum"], key=lambda row: row["target_energy"])
    assert active["ratio"] == pytest.approx(.25)


def test_measurement_validation():
    with pytest.raises(ValueError, match="matching shapes"):
        compare_measurements(np.zeros((2, 3)), np.zeros((3, 2)))
    with pytest.raises(ValueError, match="orientations"):
        compare_measurements(np.zeros((3, 3)), np.zeros((3, 3)), orientations=1)
