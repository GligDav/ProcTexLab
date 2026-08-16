import json
import numpy as np
import pytest
from procedural_texture_kernel import (FitConfig, GaborComponent, GaussianRBFComponent,
    PerlinNoiseComponent, ProceduralTextureModel, SinusoidComponent, TextureFitter,
    WaveletComponent, normalize_image)
from procedural_texture_kernel import (AnisotropicGaussianComponent, BinaryPrimitiveComponent,
    DifferenceOfGaussiansComponent, DomainWarpedNoiseComponent,
    FractalBrownianMotionComponent, LineComponent, PolynomialTrendComponent,
    RadialWaveComponent, RidgedMultifractalComponent, SparseImpulseComponent,
    SpiralWaveComponent, StepEdgeComponent, ThresholdedNoiseComponent, TurbulenceNoiseComponent,
    VoronoiNoiseComponent, SimpleConstantComponent)
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

@pytest.mark.parametrize("component", [SinusoidComponent(), GaborComponent(), GaussianRBFComponent(),
                                        PerlinNoiseComponent(), WaveletComponent()])
def test_components_are_finite(component):
    u,v=coordinate_grid(13,9); result=component.evaluate(u,v)
    assert result.shape == (9,13) and np.isfinite(result).all()

def test_model_sum_and_serialization(tmp_path):
    model=ProceduralTextureModel(.2,.1,-.2,[SinusoidComponent(.3,2,1,.4), GaussianRBFComponent(-.1,.3,.7,.12)])
    expected=model.evaluate(17,11); restored=ProceduralTextureModel.from_dict(json.loads(json.dumps(model.to_dict())))
    assert np.allclose(expected, restored.evaluate(17,11))
    path=tmp_path/"m.json"; model.save_json(path); assert np.allclose(expected, ProceduralTextureModel.load_json(path).evaluate(17,11))

def test_perlin_is_seeded_and_wavelet_is_localized():
    u, v = coordinate_grid(32, 24)
    a = PerlinNoiseComponent(seed=7, octaves=3).basis(u, v)
    b = PerlinNoiseComponent(seed=7, octaves=3).basis(u, v)
    c = PerlinNoiseComponent(seed=8, octaves=3).basis(u, v)
    assert np.array_equal(a, b) and not np.allclose(a, c)
    wavelet = WaveletComponent(center_u=.5, center_v=.5, scale_u=.08, scale_v=.08)
    values = wavelet.basis(u, v)
    assert values[12, 16] == pytest.approx(1.0)
    assert abs(values[0, 0]) < 1e-10

@pytest.mark.parametrize("component", [PerlinNoiseComponent(.2, 5, 2, .6, 2.1, .1, -.2, 12),
                                        WaveletComponent(-.3, .2, .7, .08, .15, .4)])
def test_new_component_serialization(component):
    model = ProceduralTextureModel(components=[component])
    restored = ProceduralTextureModel.from_dict(json.loads(json.dumps(model.to_dict())))
    assert type(restored.components[0]) is type(component)
    assert np.allclose(model.evaluate(19, 17), restored.evaluate(19, 17))

@pytest.mark.parametrize("component", [
    VoronoiNoiseComponent(seed=3), FractalBrownianMotionComponent(seed=3),
    RidgedMultifractalComponent(seed=3), TurbulenceNoiseComponent(seed=3),
    DomainWarpedNoiseComponent(seed=3), AnisotropicGaussianComponent(), LineComponent(),
    StepEdgeComponent(), DifferenceOfGaussiansComponent(), PolynomialTrendComponent(),
    RadialWaveComponent(), SpiralWaveComponent(), SparseImpulseComponent(seed=3),
    BinaryPrimitiveComponent(), SimpleConstantComponent(), ThresholdedNoiseComponent(seed=3),
])
def test_extended_components_are_finite_deterministic_and_serializable(component):
    u, v = coordinate_grid(23, 17)
    first = component.evaluate(u, v)
    second = component.evaluate(u, v)
    restored = ProceduralTextureModel.from_dict(
        json.loads(json.dumps(ProceduralTextureModel(components=[component]).to_dict())))
    assert first.shape == u.shape
    assert np.isfinite(first).all() and np.array_equal(first, second)
    assert type(restored.components[0]) is type(component)
    assert np.allclose(first, restored.evaluate_grid(u, v))

def test_component_variants_and_seed_changes():
    u, v = coordinate_grid(24, 20)
    assert not np.allclose(VoronoiNoiseComponent(seed=1).basis(u, v),
                           VoronoiNoiseComponent(seed=2).basis(u, v))
    assert not np.allclose(DifferenceOfGaussiansComponent(mode="dog").basis(u, v),
                           DifferenceOfGaussiansComponent(mode="log").basis(u, v))
    for shape in ("disk", "box", "ring", "checker"):
        values = BinaryPrimitiveComponent(shape=shape).basis(u, v)
        assert set(np.unique(values)) <= {0.0, 1.0}

def test_thresholded_noise_has_bounded_sharp_regions_and_serializes():
    u, v = coordinate_grid(48, 40)
    soft = ThresholdedNoiseComponent(seed=5, frequency=3, edge_width=.2)
    sharp = ThresholdedNoiseComponent(seed=5, frequency=3, edge_width=.01)
    soft_values, sharp_values = soft.basis(u, v), sharp.basis(u, v)
    assert np.min(sharp_values) >= -1 and np.max(sharp_values) <= 1
    assert np.count_nonzero(np.abs(sharp_values) > .99) > np.count_nonzero(
        np.abs(soft_values) > .99)
    restored = ProceduralTextureModel.from_dict(json.loads(json.dumps(
        ProceduralTextureModel(components=[sharp]).to_dict())))
    assert isinstance(restored.components[0], ThresholdedNoiseComponent)
    assert np.allclose(sharp.evaluate(u, v), restored.evaluate_grid(u, v))

@pytest.mark.parametrize("family, component", [
    ("perlin_noise", PerlinNoiseComponent(.2, 4, 3, seed=1)),
    ("wavelet", WaveletComponent(.3, .5, .5, .1, .1)),
    ("thresholded_noise", ThresholdedNoiseComponent(
        .25, frequency=2, octaves=4, threshold=0, edge_width=.08, seed=0)),
])
def test_new_component_families_can_be_fitted(family, component):
    image = ProceduralTextureModel(.5, components=[component]).evaluate(24, 24)
    config = FitConfig(component_families=(family,), max_components=1,
                       max_iterations=5, fitting_resolution=None, seed=0,
                       decomposition_bands=1)
    result = TextureFitter(config).fit(image)
    assert len(result.model.components) == 1
    assert result.model.components[0].type_name == family

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
                                  "autocorrelation_loss", "gradient_loss",
                                  "local_structure_loss", "mse_loss"}
    assert shifted_loss < unrelated_loss
    assert shifted_loss < np.mean((reference - shifted) ** 2)

def test_texture_loss_weight_validation():
    with pytest.raises(ValueError, match="weight"):
        TextureLossWeights(0, 0, 0, 0, 0)

def test_mse_texture_loss_component_and_weighting():
    reference = np.zeros((4, 6))
    candidate = np.full((4, 6), .5)
    total, parts = TextureLoss(reference, TextureLossWeights(0, 0, 0, 0, 1)).evaluate(candidate)
    assert parts["mse_loss"] == pytest.approx(.25)
    assert total == pytest.approx(.25)


def test_local_structure_loss_detects_phase_scrambling():
    y, x = np.indices((32, 32))
    reference = np.sign(np.sin(2 * np.pi * x / 8) + np.sin(2 * np.pi * y / 16))
    transformed = np.fft.fft2(reference)
    rng = np.random.default_rng(9)
    scrambled = np.fft.ifft2(np.abs(transformed) * np.exp(
        1j * rng.uniform(-np.pi, np.pi, transformed.shape))).real
    evaluator = TextureLoss(reference, TextureLossWeights(0, 0, 0, 0, 0, 1),
                            local_structure_scales=2,
                            local_structure_orientations=4,
                            local_structure_block_size=8)
    exact, _ = evaluator.evaluate(reference)
    changed, parts = evaluator.evaluate(scrambled)
    assert exact == pytest.approx(0.0, abs=1e-14)
    assert changed == pytest.approx(parts["local_structure_loss"])
    assert changed > 0


@pytest.mark.parametrize("setting", [
    {"local_structure_scales": 0},
    {"local_structure_orientations": 0},
    {"local_structure_block_size": 0},
])
def test_local_structure_configuration_validation(setting):
    with pytest.raises(ValueError, match="local structure"):
        TextureLoss(np.zeros((8, 8)), **setting)


def test_local_structure_candidate_limit_validation():
    with pytest.raises(ValueError, match="local_structure_candidate_limit"):
        FitConfig(local_structure_candidate_limit=0)

def test_normalization():
    assert normalize_image(np.array([[0,255]],dtype=np.uint8)).tolist() == [[0,1]]
    rgb=np.zeros((2,3,3),dtype=np.uint16); assert normalize_image(rgb).shape == (2,3)
    with pytest.raises(ValueError): normalize_image(np.zeros((2,2,2)))

def test_synthetic_fit_and_determinism():
    truth=ProceduralTextureModel(.5,components=[SinusoidComponent(.22,3,1,.35)])
    image=truth.evaluate(48,32); config=FitConfig(max_components=2,fitting_resolution=None,
                                                  max_iterations=30, decomposition_bands=1)
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


@pytest.mark.parametrize("kwargs", [
    {"detail_max_components": -1},
    {"detail_min_frequency": -1},
    {"detail_min_improvement": -1},
    {"detail_hf_ratio_threshold": 0},
    {"detail_base_sigma": 0},
    {"detail_component_families": ("unknown",)},
])
def test_detail_refinement_configuration_validation(kwargs):
    with pytest.raises(ValueError):
        FitConfig(**kwargs)


def test_detail_minimum_frequency_must_fit_enabled_frequency_range():
    with pytest.raises(ValueError, match="detail_min_frequency"):
        FitConfig(detail_refinement=True, detail_min_frequency=24, max_frequency=24)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_amplitude_refit_interval_validation(value):
    with pytest.raises(ValueError, match="amplitude_refit_interval"):
        FitConfig(amplitude_refit_interval=value)

def test_gui_has_labels_for_every_component_family():
    from gui.test_app import ATOM_LABELS
    from procedural_texture_kernel import SUPPORTED_COMPONENT_FAMILIES
    assert set(SUPPORTED_COMPONENT_FAMILIES) <= set(ATOM_LABELS)

def test_gui_defaults_to_five_decomposition_bands():
    from gui.test_app import DEFAULT_DECOMPOSITION_BANDS
    assert DEFAULT_DECOMPOSITION_BANDS == FitConfig().decomposition_bands == 5
