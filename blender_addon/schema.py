"""Strict, Blender-independent validation for PTK schema version 1."""
from __future__ import annotations

import json
import math
from pathlib import Path

SCHEMA_VERSION = 1
COORDINATE_SYSTEM = "uv_normalized_top_left_v_down_half_open"
DEFAULT_MAX_COMPONENTS = 4096
DEFAULT_MAX_SPECTRAL_MODES = 65536


class PTKImportError(ValueError):
    """A user-facing import/validation failure."""


FIELDS = {
    "sinusoid": ("amplitude", "frequency_u", "frequency_v", "phase"),
    "spectral_noise": ("amplitude", "frequencies_u", "frequencies_v", "weights", "phases"),
    "gabor": ("amplitude", "center_u", "center_v", "sigma_u", "sigma_v", "frequency", "orientation", "phase"),
    "gaussian_rbf": ("amplitude", "center_u", "center_v", "sigma"),
    "wavelet": ("amplitude", "center_u", "center_v", "scale_u", "scale_v", "orientation"),
    "anisotropic_gaussian": ("amplitude", "center_u", "center_v", "sigma_u", "sigma_v", "orientation"),
    "line": ("amplitude", "center_u", "center_v", "width", "length", "orientation", "softness"),
    "step_edge": ("amplitude", "center_u", "center_v", "orientation", "softness"),
    "dog_log": ("amplitude", "center_u", "center_v", "sigma", "ratio"),
    "polynomial_trend": ("amplitude", "linear_u", "linear_v", "quadratic_u", "cross_uv", "quadratic_v"),
    "radial_wave": ("amplitude", "center_u", "center_v", "frequency", "phase", "decay"),
    "spiral_wave": ("amplitude", "center_u", "center_v", "frequency", "phase", "decay", "arms"),
    "sparse_impulse": ("amplitude", "density", "radius"),
    "binary_primitive": ("amplitude", "center_u", "center_v", "size_u", "size_v", "orientation", "thickness"),
    "simple_constant": ("amplitude", "value"),
    "shader_graph": ("amplitude",),
    "perlin_noise": ("amplitude", "frequency", "persistence", "lacunarity", "offset_u", "offset_v"),
    "fbm": ("amplitude", "frequency", "persistence", "lacunarity", "offset_u", "offset_v"),
    "turbulence_noise": ("amplitude", "frequency", "persistence", "lacunarity", "offset_u", "offset_v"),
    "thresholded_noise": ("amplitude", "frequency", "persistence", "lacunarity", "offset_u", "offset_v", "rotation", "threshold", "edge_width"),
    "masked_noise": ("amplitude", "mask_frequency", "mask_offset_u", "mask_offset_v", "mask_rotation", "mask_threshold", "mask_edge_width", "detail_frequency", "detail_offset_u", "detail_offset_v"),
    "voronoi_noise": ("amplitude", "frequency", "jitter", "offset_u", "offset_v"),
    "ridged_multifractal": ("amplitude", "frequency", "persistence", "lacunarity", "offset_u", "offset_v", "ridge_offset", "ridge_power", "rotation", "anisotropy"),
    "domain_warped_noise": ("amplitude", "frequency", "persistence", "lacunarity", "offset_u", "offset_v", "warp_amplitude", "warp_frequency"),
    "warped_ridged_multifractal": ("amplitude", "frequency", "persistence", "lacunarity", "offset_u", "offset_v", "ridge_offset", "ridge_power", "rotation", "anisotropy", "warp_amplitude", "warp_frequency"),
    "warped_ridge_detail": ("amplitude", "ridge_frequency", "ridge_offset_u", "ridge_offset_v", "ridge_power", "ridge_rotation", "ridge_anisotropy", "warp_amplitude", "warp_frequency", "mask_threshold", "mask_edge_width", "detail_frequency", "detail_offset_u", "detail_offset_v"),
}
INTEGER_FIELDS = {"octaves", "seed", "mask_octaves", "mask_seed", "detail_octaves", "detail_seed", "ridge_octaves", "warp_octaves"}
BOOLEAN_FIELDS = {"signed", "invert_mask"}


def load_document(path: str | Path) -> dict:
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            return json.load(stream, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise PTKImportError(f"Could not read PTK JSON: {exc}") from exc


def _finite(value, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise PTKImportError(f"{path} must be a finite number")
    return float(value)


def validate_document(document: object, max_components=DEFAULT_MAX_COMPONENTS,
                      max_spectral_modes=DEFAULT_MAX_SPECTRAL_MODES) -> dict:
    if not isinstance(document, dict):
        raise PTKImportError("The JSON document must be an object")
    model = document.get("model", document)
    if not isinstance(model, dict):
        raise PTKImportError("model must be an object")
    if model.get("schema_version") != SCHEMA_VERSION:
        raise PTKImportError("model.schema_version must be exactly 1")
    if model.get("coordinate_system") != COORDINATE_SYSTEM:
        raise PTKImportError(f"coordinate_system must be {COORDINATE_SYSTEM!r}")
    for name in ("bias", "trend_u", "trend_v"):
        _finite(model.get(name, 0.0), name)
    components = model.get("components")
    if not isinstance(components, list):
        raise PTKImportError("components must be a list")
    if len(components) > max_components:
        raise PTKImportError(f"component count exceeds safety limit ({max_components})")
    modes = 0
    for index, component in enumerate(components):
        prefix = f"components[{index}]"
        if not isinstance(component, dict):
            raise PTKImportError(f"{prefix} must be an object")
        kind = component.get("type")
        if kind not in FIELDS:
            raise PTKImportError(f"{prefix}: unsupported component type {kind!r}")
        for name in FIELDS[kind]:
            if name not in component:
                raise PTKImportError(f"{prefix}.{name} is required")
            value = component[name]
            if kind == "spectral_noise" and name != "amplitude":
                if not isinstance(value, list):
                    raise PTKImportError(f"{prefix}.{name} must be a list")
                for item_index, item in enumerate(value):
                    _finite(item, f"{prefix}.{name}[{item_index}]")
            else:
                _finite(value, f"{prefix}.{name}")
        for name in INTEGER_FIELDS & component.keys():
            value = component[name]
            if isinstance(value, bool) or not isinstance(value, int) or not (-2147483648 <= value <= 2147483647):
                raise PTKImportError(f"{prefix}.{name} must be a safe integer")
            if "octaves" in name and not 1 <= value <= 16:
                raise PTKImportError(f"{prefix}.{name} must be between 1 and 16")
        for name in BOOLEAN_FIELDS & component.keys():
            if not isinstance(component[name], bool):
                raise PTKImportError(f"{prefix}.{name} must be boolean")
        if kind == "spectral_noise":
            lengths = {len(component[name]) for name in ("frequencies_u", "frequencies_v", "weights", "phases")}
            if len(lengths) != 1:
                raise PTKImportError(f"{prefix}: spectral arrays must have equal lengths")
            modes += lengths.pop()
        if kind == "shader_graph":
            graph = component.get("graph")
            if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list):
                raise PTKImportError(f"{prefix}.graph.nodes must be a list")
            nodes = graph["nodes"]
            if not 1 <= len(nodes) <= 64:
                raise PTKImportError(f"{prefix}.graph must contain between 1 and 64 nodes")
            seen = set()
            arity = {"component": 0, "constant": 0, "one_minus": 1,
                     "smoothstep": 1, "add": 2, "multiply": 2, "mix": 3}
            for graph_index, graph_node in enumerate(nodes):
                node_path = f"{prefix}.graph.nodes[{graph_index}]"
                if not isinstance(graph_node, dict):
                    raise PTKImportError(f"{node_path} must be an object")
                node_id, operation = graph_node.get("id"), graph_node.get("operation")
                inputs = graph_node.get("inputs", [])
                if not isinstance(node_id, str) or not node_id or node_id in seen:
                    raise PTKImportError(f"{node_path}.id must be non-empty and unique")
                if operation not in arity:
                    raise PTKImportError(f"{node_path}.operation is unsupported")
                if not isinstance(inputs, list) or len(inputs) != arity[operation]:
                    raise PTKImportError(f"{node_path}.inputs has invalid arity")
                if any(source not in seen for source in inputs):
                    raise PTKImportError(f"{node_path}.inputs must reference earlier nodes")
                if operation == "component":
                    embedded = graph_node.get("component")
                    nested = {"schema_version": 1, "coordinate_system": COORDINATE_SYSTEM,
                              "components": [embedded]}
                    validate_document(nested, max_components=max_components,
                                      max_spectral_modes=max_spectral_modes)
                elif operation == "constant":
                    _finite(graph_node.get("value", 0.0), f"{node_path}.value")
                elif operation == "smoothstep":
                    edge0 = _finite(graph_node.get("edge0", -0.08), f"{node_path}.edge0")
                    edge1 = _finite(graph_node.get("edge1", 0.08), f"{node_path}.edge1")
                    if edge1 <= edge0:
                        raise PTKImportError(f"{node_path}.edge1 must exceed edge0")
                seen.add(node_id)
            if graph.get("output_node") not in seen:
                raise PTKImportError(f"{prefix}.graph.output_node does not exist")
        if kind == "dog_log" and component.get("mode", "dog") not in {"dog", "log"}:
            raise PTKImportError(f"{prefix}.mode must be 'dog' or 'log'")
        if kind == "binary_primitive" and component.get("shape") not in {"disk", "box", "ring", "checker"}:
            raise PTKImportError(f"{prefix}.shape is invalid")
    if modes > max_spectral_modes:
        raise PTKImportError(f"spectral mode count exceeds safety limit ({max_spectral_modes})")
    return model


def metric_summary(document: dict) -> str:
    metrics = document.get("metrics", {})
    if not isinstance(metrics, dict):
        return ""
    pairs = []
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            pairs.append(f"{key}={value:.6g}")
    return ", ".join(pairs)[:2048]
