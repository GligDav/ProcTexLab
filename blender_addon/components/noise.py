"""Compact, explicitly approximate native Blender noise translations."""
from __future__ import annotations

from ..compatibility import socket


def _vector(tree, n, u, v, scale=1.0, offset_u=0.0, offset_v=0.0):
    combine = tree.nodes.new("ShaderNodeCombineXYZ")
    n.links.new(n.add(n.mul(u, scale), offset_u), combine.inputs["X"])
    n.links.new(n.add(n.mul(v, scale), offset_v), combine.inputs["Y"])
    return combine.outputs["Vector"]


def native_noise(tree, n, u, v, c):
    node = tree.nodes.new("ShaderNodeTexNoise")
    node.noise_dimensions = "3D"
    vec = _vector(tree, n, u, v, c.get("frequency", c.get("detail_frequency", 4.0)), c.get("offset_u", 0) + c.get("seed", 0) * 17.17, c.get("offset_v", 0))
    tree.links.new(vec, node.inputs["Vector"])
    node.inputs["Scale"].default_value = 1.0
    node.inputs["Detail"].default_value = max(0.0, float(c.get("octaves", c.get("detail_octaves", 2))) - 1.0)
    node.inputs["Roughness"].default_value = min(max(float(c.get("persistence", .5)), 0), 1)
    signed = n.sub(n.mul(2, node.outputs["Fac"]), 1)
    if c["type"] == "turbulence_noise": signed = n.sub(n.mul(2, n.math("ABSOLUTE", signed)), 1)
    return n.mul(c["amplitude"], signed)


def voronoi(tree, n, u, v, c):
    node = tree.nodes.new("ShaderNodeTexVoronoi"); node.distance = "EUCLIDEAN"; node.feature = "F1"
    tree.links.new(_vector(tree, n, u, v, c["frequency"], c["offset_u"] + c.get("seed", 0) * 11.3, c["offset_v"]), node.inputs["Vector"])
    node.inputs["Scale"].default_value = 1.0; node.inputs["Randomness"].default_value = c["jitter"]
    basis = n.math("CLAMP", n.sub(1, n.mul(2 ** .5, node.outputs["Distance"])))
    return n.mul(c["amplitude"], basis)


def generic(tree, n, u, v, c):
    # Composite noise families retain their dominant scale/range in compact mode.
    return native_noise(tree, n, u, v, c)


APPROXIMATE_BUILDERS = {name: generic for name in (
    "perlin_noise", "fbm", "turbulence_noise", "thresholded_noise", "masked_noise",
    "ridged_multifractal", "domain_warped_noise", "warped_ridged_multifractal", "warped_ridge_detail",
    "sparse_impulse",
)}
APPROXIMATE_BUILDERS["voronoi_noise"] = voronoi
