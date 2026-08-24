"""Schema tests run in normal Python without importing bpy."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("ptk_blender_schema", Path(__file__).parents[1] / "schema.py")
schema = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(schema)


def model(components=None):
    return {"schema_version": 1, "coordinate_system": schema.COORDINATE_SYSTEM,
            "bias": 0.0, "trend_u": 0.0, "trend_v": 0.0, "components": components or []}


def test_accepts_envelope_and_bare_model():
    value = model()
    assert schema.validate_document(value) is value
    assert schema.validate_document({"schema_version": 1, "model": value}) is value


@pytest.mark.parametrize("document", [None, [], {"model": []}, model() | {"schema_version": 2},
    model() | {"coordinate_system": "wrong"}, model() | {"bias": float("nan")}, model() | {"components": {}}])
def test_rejects_invalid_documents(document):
    with pytest.raises(schema.PTKImportError): schema.validate_document(document)


def test_spectral_arrays_must_match_and_be_finite():
    component = {"type": "spectral_noise", "amplitude": 1, "frequencies_u": [1],
                 "frequencies_v": [2], "weights": [1], "phases": []}
    with pytest.raises(schema.PTKImportError, match="equal lengths"):
        schema.validate_document(model([component]))
    component["phases"] = [float("inf")]
    with pytest.raises(schema.PTKImportError, match="finite"):
        schema.validate_document(model([component]))


def test_unknown_component_never_silently_skips():
    with pytest.raises(schema.PTKImportError, match="unsupported component"):
        schema.validate_document(model([{"type": "future_atom"}]))


def test_example_result_validates():
    document = schema.load_document(Path(__file__).parents[1] / "example_result.json")
    validated = schema.validate_document(document)
    assert validated["components"]


def test_shader_graph_validates_embedded_components():
    component = {"type": "shader_graph", "amplitude": 0.5, "graph": {
        "nodes": [
            {"id": "wave", "operation": "component", "inputs": [], "component": {
                "type": "sinusoid", "amplitude": 1.0, "frequency_u": 2.0,
                "frequency_v": 0.0, "phase": 0.0}},
            {"id": "mask", "operation": "smoothstep", "inputs": ["wave"],
             "edge0": -0.1, "edge1": 0.1},
        ], "output_node": "mask"}}
    assert schema.validate_document(model([component]))["components"][0] is component


def test_shader_graph_rejects_forward_references():
    component = {"type": "shader_graph", "amplitude": 1.0, "graph": {
        "nodes": [{"id": "bad", "operation": "one_minus", "inputs": ["later"]}],
        "output_node": "bad"}}
    with pytest.raises(schema.PTKImportError, match="earlier nodes"):
        schema.validate_document(model([component]))
