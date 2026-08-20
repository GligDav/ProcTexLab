import json
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from procedural_texture_kernel import (FitConfig, GaborComponent, GaussianRBFComponent,
    PerlinNoiseComponent, ProceduralTextureModel, SinusoidComponent,
    SpectralNoiseComponent, TextureFitter,
    WaveletComponent, normalize_image)
from procedural_texture_kernel import (AnisotropicGaussianComponent, BinaryPrimitiveComponent,
    DifferenceOfGaussiansComponent, DomainWarpedNoiseComponent,
    FractalBrownianMotionComponent, LineComponent, MaskedNoiseComponent, PolynomialTrendComponent,
    RadialWaveComponent, RidgedMultifractalComponent, SparseImpulseComponent,
    SpiralWaveComponent, StepEdgeComponent, ThresholdedNoiseComponent, TurbulenceNoiseComponent,
    VoronoiNoiseComponent, WarpedRidgeDetailComponent,
    WarpedRidgedMultifractalComponent,
    SimpleConstantComponent)
from procedural_texture_kernel.coordinates import coordinate_grid, coordinate_grid_region
from procedural_texture_kernel.metrics import calculate_metrics
from procedural_texture_kernel.fitting import (
    _adaptive_noise_frequencies, _diverse_candidate_shortlist,
    _local_structure_estimate,
    _perlin_candidates, _refine_model_parameters, _refine_new_atom,
    _spectral_noise_candidate)
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
                                        SpectralNoiseComponent(
                                            frequencies_u=(2., 4.),
                                            frequencies_v=(1., -3.),
                                            weights=(1., .4), phases=(.2, -.7)),
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


def test_spectral_noise_candidate_captures_multiple_residual_modes():
    u, v = coordinate_grid(48, 40)
    residual = (.8 * np.cos(2 * np.pi * (7 * u + 2 * v) + .3)
                + .35 * np.cos(2 * np.pi * (-3 * u + 6 * v) - .8))
    candidate = _spectral_noise_candidate(
        residual, FitConfig(spectral_noise_modes=2, max_frequency=12))
    assert candidate is not None
    assert len(candidate.weights) == 2
    basis = candidate.basis(u, v)
    amplitude = np.vdot(basis, residual).real / np.vdot(basis, basis).real
    assert np.mean((residual - amplitude * basis) ** 2) < np.mean(residual ** 2) * .2
    assert np.allclose(basis, candidate.basis(u + 1, v + 1), atol=1e-12)


def test_spectral_noise_rejects_mismatched_mode_arrays():
    component = SpectralNoiseComponent(
        frequencies_u=(1.,), frequencies_v=(1., 2.),
        weights=(1.,), phases=(0.,))
    with pytest.raises(ValueError, match="equal lengths"):
        component.basis(*coordinate_grid(8, 8))

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
    WarpedRidgedMultifractalComponent(seed=3),
    WarpedRidgeDetailComponent(mask_seed=3, detail_seed=9),
    StepEdgeComponent(), DifferenceOfGaussiansComponent(), PolynomialTrendComponent(),
    RadialWaveComponent(), SpiralWaveComponent(), SparseImpulseComponent(seed=3),
    BinaryPrimitiveComponent(), SimpleConstantComponent(), ThresholdedNoiseComponent(seed=3),
    MaskedNoiseComponent(mask_seed=3, detail_seed=7),
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
    isotropic = RidgedMultifractalComponent(seed=4).basis(u, v)
    directional = RidgedMultifractalComponent(
        seed=4, rotation=.6, anisotropy=2.5, ridge_power=4).basis(u, v)
    assert np.min(directional) >= -1 and np.max(directional) <= 1
    assert not np.allclose(isotropic, directional)
    warped = WarpedRidgedMultifractalComponent(
        seed=4, rotation=.6, anisotropy=2.5, warp_amplitude=.3).basis(u, v)
    assert not np.allclose(directional, warped)
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

def test_masked_noise_complement_partitions_the_same_detail_field():
    u, v = coordinate_grid(40, 32)
    normal = MaskedNoiseComponent(mask_seed=3, detail_seed=9, invert_mask=False)
    inverse = MaskedNoiseComponent(mask_seed=3, detail_seed=9, invert_mask=True)
    detail = PerlinNoiseComponent(
        frequency=normal.detail_frequency, octaves=normal.detail_octaves,
        seed=normal.detail_seed).basis(u, v)
    assert np.allclose(normal.basis(u, v) + inverse.basis(u, v), detail)
    assert not np.allclose(normal.basis(u, v), inverse.basis(u, v))


def test_warped_ridge_detail_complements_share_the_same_detail():
    u, v = coordinate_grid(40, 32)
    settings = dict(mask_seed=3, detail_seed=9, ridge_rotation=.4,
                    ridge_anisotropy=2.0, warp_amplitude=.2)
    normal = WarpedRidgeDetailComponent(**settings, invert_mask=False)
    inverse = WarpedRidgeDetailComponent(**settings, invert_mask=True)
    detail = PerlinNoiseComponent(
        frequency=normal.detail_frequency, octaves=normal.detail_octaves,
        seed=normal.detail_seed).basis(u, v)
    assert np.allclose(normal.basis(u, v) + inverse.basis(u, v), detail)
    assert not np.allclose(normal.basis(u, v), inverse.basis(u, v))

def test_new_component_families_can_be_fitted():
    cases = (
        ("spectral_noise", SpectralNoiseComponent(
            .2, frequencies_u=(4., 7.), frequencies_v=(1., -2.),
            weights=(1., .35), phases=(.2, -.6))),
        ("perlin_noise", PerlinNoiseComponent(.2, 4, 3, seed=1)),
        ("wavelet", WaveletComponent(.3, .5, .5, .1, .1)),
        ("thresholded_noise", ThresholdedNoiseComponent(
            .25, frequency=2, octaves=4, threshold=0, edge_width=.08, seed=0)),
        ("ridged_multifractal", RidgedMultifractalComponent(
            .2, frequency=2, octaves=4, ridge_power=3, seed=0)),
        ("warped_ridged_multifractal", WarpedRidgedMultifractalComponent(
            .2, frequency=2, octaves=4, ridge_power=3,
            warp_amplitude=.2, seed=0)),
        ("masked_noise", MaskedNoiseComponent(
            .2, mask_frequency=2, mask_seed=0, detail_frequency=8,
            detail_seed=1009)),
    )
    for family, component in cases:
        image = ProceduralTextureModel(
            .5, components=[component]).evaluate(24, 24)
        config = FitConfig(component_families=(family,), max_components=1,
                           max_iterations=5, fitting_resolution=None, seed=0,
                           decomposition_bands=1)
        result = TextureFitter(config).fit(image)
        assert len(result.model.components) == 1, family
        assert result.model.components[0].type_name == family

@pytest.mark.parametrize("initial,target_atom", [
    (AnisotropicGaussianComponent(.4, .35, .45, .12, .06, .2),
     AnisotropicGaussianComponent(.4, .55, .5, .15, .08, .4)),
    (LineComponent(.4, .35, .45, .08, .7, .2, .03),
     LineComponent(.4, .55, .5, .05, .9, .4, .015)),
    (StepEdgeComponent(.4, .35, .45, .2, .04),
     StepEdgeComponent(.4, .55, .5, .4, .02)),
    (DifferenceOfGaussiansComponent(.4, .35, .45, .1, 1.6),
     DifferenceOfGaussiansComponent(.4, .55, .5, .13, 2.0)),
    (VoronoiNoiseComponent(.4, 3, .8, .1, -.1, 2),
     VoronoiNoiseComponent(.4, 4, 1.0, .2, -.2, 2)),
    (FractalBrownianMotionComponent(.4, 3, 4, .5, 2, .1, -.1, 2),
     FractalBrownianMotionComponent(.4, 4, 4, .6, 2.2, .2, -.2, 2)),
    (DomainWarpedNoiseComponent(.4, 3, 4, .5, 2, .1, -.1, 2, .1, 2),
     DomainWarpedNoiseComponent(.4, 4, 4, .6, 2.2, .2, -.2, 2, .2, 3)),
])
def test_structural_atom_refinement_never_worsens_objective(initial, target_atom):
    u, v = coordinate_grid(20, 18)
    target = target_atom.evaluate(u, v)
    loss = TextureLoss(target, TextureLossWeights(0, 0, 0, 0, 1))
    current = np.zeros_like(target)
    before, _ = loss.evaluate(current + initial.evaluate(u, v))
    refined = _refine_new_atom(initial, current, loss, u, v, 8)
    after, _ = loss.evaluate(current + refined.evaluate(u, v))
    assert type(refined) is type(initial)
    assert after <= before + 1e-15

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
    assert set(shifted_parts) == {"texture_loss", "spectrum_loss",
                                  "absolute_spectrum_loss", "histogram_loss",
                                  "oriented_spectrum_loss",
                                  "autocorrelation_loss", "gradient_loss",
                                  "local_structure_loss", "local_contrast_loss", "mse_loss"}
    assert shifted_loss < unrelated_loss
    assert shifted_loss < np.mean((reference - shifted) ** 2)


@pytest.mark.parametrize("weights", [
    TextureLossWeights(0, 0, 0, 0, 1),
    TextureLossWeights(1, .5, .75, .5, 1, 0, 0, .25, .25),
    TextureLossWeights(0, 0, 0, 0, 0, 1, 1),
])
def test_weighted_only_texture_loss_matches_full_evaluation(weights):
    rng = np.random.default_rng(123)
    reference = rng.normal(size=(24, 20))
    candidate = rng.normal(size=(24, 20))
    loss = TextureLoss(reference, weights)
    full, _ = loss.evaluate(candidate)
    assert loss.evaluate_total(candidate) == pytest.approx(full, abs=1e-15)

def test_texture_loss_weight_validation():
    with pytest.raises(ValueError, match="weight"):
        TextureLossWeights(0, 0, 0, 0, 0)

def test_mse_texture_loss_component_and_weighting():
    reference = np.zeros((4, 6))
    candidate = np.full((4, 6), .5)
    total, parts = TextureLoss(reference, TextureLossWeights(0, 0, 0, 0, 1)).evaluate(candidate)
    assert parts["mse_loss"] == pytest.approx(.25)
    assert total == pytest.approx(.25)


def test_absolute_spectrum_loss_detects_missing_energy():
    y, x = np.indices((48, 48))
    reference = np.sin(2 * np.pi * x / 4)
    weights = TextureLossWeights(0, 0, 0, 0, 0, 0, 0,
                                 absolute_spectrum=1)
    evaluator = TextureLoss(reference, weights)
    exact, _ = evaluator.evaluate(reference)
    attenuated, parts = evaluator.evaluate(reference * .5)
    assert exact == pytest.approx(0.0, abs=1e-14)
    assert attenuated == pytest.approx(parts["absolute_spectrum_loss"])
    assert attenuated > 0


def test_oriented_spectrum_loss_detects_wrong_direction():
    y, x = np.indices((48, 48))
    reference = np.sin(2 * np.pi * x / 6)
    candidate = np.sin(2 * np.pi * y / 6)
    weights = TextureLossWeights(0, 0, 0, 0, 0, 0, 0, 0,
                                 oriented_spectrum=1)
    evaluator = TextureLoss(reference, weights)
    exact, _ = evaluator.evaluate(reference)
    changed, parts = evaluator.evaluate(candidate)
    assert exact == pytest.approx(0.0, abs=1e-14)
    assert changed == pytest.approx(parts["oriented_spectrum_loss"])
    assert changed > 0

def test_local_contrast_loss_detects_region_scale_changes():
    y, x = np.indices((48, 48))
    reference = (x >= 24).astype(float)
    blurred = gaussian_filter(reference, 3.0)
    weights = TextureLossWeights(0, 0, 0, 0, 0, 0, 1)
    evaluator = TextureLoss(reference, weights)
    exact, _ = evaluator.evaluate(reference)
    changed, parts = evaluator.evaluate(blurred)
    assert exact == pytest.approx(0.0, abs=1e-14)
    assert changed == pytest.approx(parts["local_contrast_loss"])
    assert changed > 0


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


def test_automatic_frequency_limit_is_resolved_from_fit_shape():
    image = np.zeros((40, 80))
    result = TextureFitter(FitConfig(max_components=0, fitting_resolution=None)).fit(image)
    frequency = result.metadata["frequency_range"]
    assert frequency["maximum_mode"] == "automatic"
    assert frequency["maximum"] == pytest.approx(18.0)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_amplitude_refit_interval_validation(value):
    with pytest.raises(ValueError, match="amplitude_refit_interval"):
        FitConfig(amplitude_refit_interval=value)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_parameter_refinement_pass_validation(value):
    with pytest.raises(ValueError, match="parameter_refinement_passes"):
        FitConfig(parameter_refinement_passes=value)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_parameter_refinement_atom_limit_validation(value):
    with pytest.raises(ValueError, match="parameter_refinement_atom_limit"):
        FitConfig(parameter_refinement_atom_limit=value)


def test_joint_parameter_refinement_never_worsens_band_loss():
    u, v = coordinate_grid(32, 32)
    target = SinusoidComponent(.4, 5, 2, .7).evaluate(u, v)
    model = ProceduralTextureModel(components=[SinusoidComponent(.3, 4.5, 2.3, .2)])
    loss = TextureLoss(target, TextureLossWeights(0, 0, 0, 0, 1))
    before, _ = loss.evaluate(model.evaluate_grid(u, v))
    metadata = _refine_model_parameters(
        model, target, loss, u, v,
        FitConfig(max_iterations=20, max_frequency=12,
                  parameter_refinement_passes=1), None, None)
    after, _ = loss.evaluate(model.evaluate_grid(u, v))
    assert metadata["attempted"]
    assert after <= before + 1e-12

def test_band_aware_candidates_validation():
    with pytest.raises(ValueError, match="band_aware_candidates"):
        FitConfig(band_aware_candidates=1)

@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_noise_seed_candidate_validation(value):
    with pytest.raises(ValueError, match="noise_seed_candidates"):
        FitConfig(noise_seed_candidates=value)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_spectral_noise_mode_validation(value):
    with pytest.raises(ValueError, match="spectral_noise_modes"):
        FitConfig(spectral_noise_modes=value)

def test_adaptive_noise_frequencies_include_residual_peak():
    y, x = np.indices((40, 48))
    residual = np.sin(2 * np.pi * (7 * x / 48 + 2 * y / 40))
    frequencies = _adaptive_noise_frequencies(residual, FitConfig())
    assert len(frequencies) <= 3
    assert any(abs(frequency - np.hypot(7, 2)) < .5 for frequency in frequencies)

def test_local_structure_estimate_detects_vertical_edge_normal():
    residual = np.zeros((48, 48))
    residual[:, 24:] = 1
    normal, tangent, scale = _local_structure_estimate(residual, 24, 24)
    assert abs(np.sin(normal)) < .2
    assert abs(np.cos(tangent)) < .2
    assert .025 <= scale <= .25

def test_noise_candidate_seed_bank_is_deterministic_and_bounded():
    residual = np.random.default_rng(3).normal(size=(20, 24))
    u, v = coordinate_grid(24, 20)
    config = FitConfig(component_families=("perlin_noise",),
                       noise_seed_candidates=2)
    first = _perlin_candidates(residual, config, u, v)
    second = _perlin_candidates(residual, config, u, v)
    assert 1 <= len(first) <= 6
    assert {atom.seed for atom in first} == {0, 1}
    assert [atom.to_dict() for atom in first] == [atom.to_dict() for atom in second]


def test_candidate_shortlist_preserves_family_diversity():
    candidates = [(10.0, SinusoidComponent()),
                  (9.0, SinusoidComponent(frequency_u=2)),
                  (1.0, PerlinNoiseComponent()),
                  (.5, WaveletComponent())]
    selected = _diverse_candidate_shortlist(candidates, 3)
    assert {atom.type_name for _, atom in selected} == {
        "sinusoid", "perlin_noise", "wavelet"}

def test_masked_noise_candidates_cover_both_regions():
    y, x = np.indices((24, 28))
    residual = np.where(x < 10, .5, -.2) + .05 * np.sin(2*np.pi*y/5)
    u, v = coordinate_grid(28, 24)
    config = FitConfig(component_families=("masked_noise",),
                       noise_seed_candidates=1)
    candidates = _perlin_candidates(residual, config, u, v)
    assert candidates
    assert {atom.invert_mask for atom in candidates} == {False, True}
    assert all(isinstance(atom, MaskedNoiseComponent) for atom in candidates)


def test_warped_ridge_detail_candidates_cover_both_regions():
    residual = np.random.default_rng(4).normal(size=(20, 24))
    u, v = coordinate_grid(24, 20)
    config = FitConfig(component_families=("warped_ridge_detail",),
                       noise_seed_candidates=1)
    candidates = _perlin_candidates(residual, config, u, v)
    assert candidates
    assert {atom.invert_mask for atom in candidates} == {False, True}
    assert all(isinstance(atom, WarpedRidgeDetailComponent)
               for atom in candidates)

def test_gui_has_labels_for_every_component_family():
    from gui.test_app import ATOM_LABELS
    from procedural_texture_kernel import SUPPORTED_COMPONENT_FAMILIES
    assert set(SUPPORTED_COMPONENT_FAMILIES) <= set(ATOM_LABELS)

def test_gui_defaults_to_five_decomposition_bands():
    from gui.test_app import DEFAULT_DECOMPOSITION_BANDS
    assert DEFAULT_DECOMPOSITION_BANDS == FitConfig().decomposition_bands == 5
