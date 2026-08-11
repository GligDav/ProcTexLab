import json
import numpy as np
import pytest
from procedural_texture_kernel import (FitConfig, GaborComponent, GaussianRBFComponent,
    ProceduralTextureModel, SinusoidComponent, TextureFitter, normalize_image)
from procedural_texture_kernel.coordinates import coordinate_grid, coordinate_grid_region
from procedural_texture_kernel.metrics import calculate_metrics
from procedural_texture_kernel.texture_loss import TextureLoss, TextureLossWeights

def test_coordinates():
    u, v = coordinate_grid(4, 2); assert u.shape == (2,4); assert v.shape == (2,4)
    assert u[0,-1] == .75 and v[-1,0] == .5

def test_coordinate_region_and_extended_evaluation():
    u, v = coordinate_grid_region(4, 2, (1, 3), (-1, 1))
    assert np.allclose(u[0], [1, 1.5, 2, 2.5])
    assert np.allclose(v[:, 0], [-1, 0])
    model = ProceduralTextureModel(components=[SinusoidComponent(1, 1, 0, 0)])
    regular = model.evaluate(8, 5)
    extended = model.evaluate_region(16, 5, (0, 2), (0, 1))
    assert np.allclose(extended[:, :8], regular)
    assert np.allclose(extended[:, 8:], regular)
    with pytest.raises(ValueError, match="upper bound"):
        model.evaluate_region(4, 4, (1, 1), (0, 1))

@pytest.mark.parametrize("component", [SinusoidComponent(), GaborComponent(), GaussianRBFComponent()])
def test_components_are_finite(component):
    u,v=coordinate_grid(13,9); result=component.evaluate(u,v)
    assert result.shape == (9,13) and np.isfinite(result).all()

def test_model_sum_and_serialization(tmp_path):
    model=ProceduralTextureModel(.2,.1,-.2,[SinusoidComponent(.3,2,1,.4), GaussianRBFComponent(-.1,.3,.7,.12)])
    expected=model.evaluate(17,11); restored=ProceduralTextureModel.from_dict(json.loads(json.dumps(model.to_dict())))
    assert np.allclose(expected, restored.evaluate(17,11))
    path=tmp_path/"m.json"; model.save_json(path); assert np.allclose(expected, ProceduralTextureModel.load_json(path).evaluate(17,11))

def test_metrics_known():
    m=calculate_metrics(np.array([0.,1.]),np.array([0.,0.]))
    assert m["mse"] == .5 and m["rmse"] == pytest.approx(np.sqrt(.5)) and m["mae"] == .5

def test_texture_loss_is_statistical_and_reports_components():
    rng = np.random.default_rng(4)
    reference = rng.random((32, 40))
    shifted = np.roll(reference, (5, 7), axis=(0, 1))
    evaluator = TextureLoss(reference)
    exact, exact_parts = evaluator.evaluate(reference)
    shifted_loss, shifted_parts = evaluator.evaluate(shifted)
    unrelated_loss, _ = evaluator.evaluate(rng.random(reference.shape))
    assert exact == pytest.approx(0.0, abs=1e-14)
    assert set(shifted_parts) == {"texture_loss", "spectrum_loss", "histogram_loss",
                                  "autocorrelation_loss", "gradient_loss"}
    assert shifted_loss < unrelated_loss
    assert shifted_loss < np.mean((reference - shifted) ** 2)

def test_texture_loss_weight_validation():
    with pytest.raises(ValueError, match="weight"):
        TextureLossWeights(0, 0, 0, 0)

def test_normalization():
    assert normalize_image(np.array([[0,255]],dtype=np.uint8)).tolist() == [[0,1]]
    rgb=np.zeros((2,3,3),dtype=np.uint16); assert normalize_image(rgb).shape == (2,3)
    with pytest.raises(ValueError): normalize_image(np.zeros((2,2,2)))

def test_synthetic_fit_and_determinism():
    truth=ProceduralTextureModel(.5,components=[SinusoidComponent(.22,3,1,.35)])
    image=truth.evaluate(48,32); config=FitConfig(max_components=2,fitting_resolution=None,max_iterations=30)
    a=TextureFitter(config).fit(image); b=TextureFitter(config).fit(image)
    assert a.metrics["rmse"] < .015
    assert np.allclose(a.reconstruction,b.reconstruction)

def test_constant_and_non_square():
    result=TextureFitter(FitConfig(max_components=2)).fit(np.full((19,31),.37))
    assert result.evaluate(31,19).shape == (19,31) and result.metrics["rmse"] < 1e-7

@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_min_improvement_validation(value):
    with pytest.raises(ValueError, match="min_improvement"):
        FitConfig(min_improvement=value)
