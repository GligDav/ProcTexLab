import json

import numpy as np
import pytest

from procedural_texture_kernel import (
    FitConfig, PerlinNoiseComponent, ProceduralTextureModel,
    ShaderGraph, ShaderGraphComponent, ShaderNode,
)
from procedural_texture_kernel.coordinates import coordinate_grid
from procedural_texture_kernel.fitting import _perlin_candidates
from procedural_texture_kernel.fitting import _refine_new_atom
from procedural_texture_kernel.texture_loss import TextureLoss, TextureLossWeights


def test_shader_graph_operations_and_model_round_trip():
    graph = ShaderGraph([
        ShaderNode("zero", "constant", value=0),
        ShaderNode("one", "constant", value=1),
        ShaderNode("factor", "constant", value=.25),
        ShaderNode("mixed", "mix", ("zero", "one", "factor")),
        ShaderNode("inverted", "one_minus", ("mixed",)),
        ShaderNode("result", "multiply", ("mixed", "inverted")),
    ], "result")
    model = ProceduralTextureModel(.1, components=[ShaderGraphComponent(2, graph)])
    expected = model.evaluate(13, 9)
    assert np.allclose(expected, .475)
    restored = ProceduralTextureModel.from_dict(json.loads(json.dumps(model.to_dict())))
    assert isinstance(restored.components[0], ShaderGraphComponent)
    assert np.allclose(expected, restored.evaluate(13, 9))


def test_shader_graph_smoothstep_and_component_source():
    source = PerlinNoiseComponent(frequency=3, octaves=2, seed=4)
    graph = ShaderGraph([
        ShaderNode("source", "component", component=source),
        ShaderNode("mask", "smoothstep", ("source",), edge0=-.05, edge1=.05),
    ], "mask")
    u, v = coordinate_grid(20, 16)
    values = graph.evaluate(u, v)
    assert values.shape == u.shape
    assert np.min(values) >= 0 and np.max(values) <= 1


@pytest.mark.parametrize("nodes,output,match", [
    ([ShaderNode("x", "unknown")], "x", "unsupported"),
    ([ShaderNode("sum", "add", ("missing", "missing"))], "sum", "earlier"),
    ([ShaderNode("x", "constant")], "missing", "output"),
    ([ShaderNode("x", "smoothstep", ("missing",), edge0=1, edge1=0)], "x", "earlier"),
])
def test_shader_graph_validation(nodes, output, match):
    with pytest.raises(ValueError, match=match):
        ShaderGraph(nodes, output)


def test_fitter_proposes_bounded_region_mix_graphs():
    y, x = np.indices((24, 28))
    residual = np.where(x < 12, .4, -.2) + .08*np.sin(2*np.pi*y/5)
    u, v = coordinate_grid(28, 24)
    config = FitConfig(component_families=("shader_graph",),
                       noise_seed_candidates=1)
    candidates = _perlin_candidates(residual, config, u, v)
    assert 1 <= len(candidates) <= 3
    assert all(isinstance(candidate, ShaderGraphComponent) for candidate in candidates)
    assert all(len(candidate.graph.nodes) == 5 for candidate in candidates)
    assert all(np.isfinite(candidate.basis(u, v)).all() for candidate in candidates)


def test_region_mix_graph_refinement_preserves_topology_and_never_worsens():
    y, x = np.indices((20, 24))
    residual = np.where(x < 10, .4, -.2) + .06*np.sin(2*np.pi*y/4)
    u, v = coordinate_grid(24, 20)
    config = FitConfig(component_families=("shader_graph",),
                       noise_seed_candidates=1)
    initial = _perlin_candidates(residual, config, u, v)[0]
    target = np.roll(initial.basis(u, v), 2, axis=1)
    loss = TextureLoss(target, TextureLossWeights(0, 0, 0, 0, 1))
    before, _ = loss.evaluate(initial.evaluate(u, v))
    refined = _refine_new_atom(initial, np.zeros_like(target), loss, u, v, 5)
    after, _ = loss.evaluate(refined.evaluate(u, v))
    assert isinstance(refined, ShaderGraphComponent)
    assert [node.operation for node in refined.graph.nodes] == [
        "component", "smoothstep", "component", "component", "mix"]
    assert after <= before + 1e-15
