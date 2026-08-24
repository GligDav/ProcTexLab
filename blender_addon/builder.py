"""Transactional PTK model-group and material construction."""
from __future__ import annotations

import json
from pathlib import Path

import bpy

from .compatibility import add_group_socket, principled_input
from .components import APPROXIMATE_TYPES, BUILDERS
from .node_utils import Nodes, tag_component
from .schema import PTKImportError, metric_summary

GROUP_TAG = "ptk_model_group"
NODE_TAG = "ptk_model_node"


def estimate_node_count(model, approximate=False):
    total = 12
    for c in model["components"]:
        if c["type"] == "shader_graph":
            nested = [node["component"] for node in c["graph"]["nodes"] if node["operation"] == "component"]
            total += len(c["graph"]["nodes"]) * 4 + estimate_node_count({"components": nested}, approximate)
        elif c["type"] == "spectral_noise": total += 7 * len(c["weights"]) + 3
        elif c["type"] in APPROXIMATE_TYPES: total += 8 if approximate else 1000
        else: total += 24
    return total


def _approximate_components(components, prefix=""):
    result = []
    for index, component in enumerate(components):
        path = f"{prefix}{index}"
        if component["type"] in APPROXIMATE_TYPES:
            result.append((path, component["type"]))
        elif component["type"] == "shader_graph":
            embedded = [node["component"] for node in component["graph"]["nodes"] if node["operation"] == "component"]
            result.extend(_approximate_components(embedded, path + ".graph.component."))
    return result


def build_model_group(model, name, allow_approximate=False, node_limit=100000, progress=None):
    approximate = _approximate_components(model["components"])
    if approximate and not allow_approximate:
        details = ", ".join(f"{kind} at {path}" for path, kind in approximate[:8])
        raise PTKImportError(f"Exact node translation is unavailable for: {details}. Enable Compact (Approximate) explicitly.")
    estimate = estimate_node_count(model, allow_approximate)
    if estimate > node_limit:
        raise PTKImportError(f"Estimated {estimate:,} nodes exceed the configured limit of {node_limit:,}")
    tree = bpy.data.node_groups.new(name, "ShaderNodeTree")
    tree[GROUP_TAG] = True; tree["ptk_schema_version"] = 1
    try:
        add_group_socket(tree, "Fitted Coordinates", "INPUT", "NodeSocketVector")
        add_group_socket(tree, "Value", "OUTPUT", "NodeSocketFloat")
        inputs = tree.nodes.new("NodeGroupInput"); inputs.location = (-1000, 0)
        outputs = tree.nodes.new("NodeGroupOutput"); outputs.location = (1200, 0)
        sep = tree.nodes.new("ShaderNodeSeparateXYZ")
        tree.links.new(inputs.outputs["Fitted Coordinates"], sep.inputs["Vector"])
        n = Nodes(tree); u, v = sep.outputs["X"], sep.outputs["Y"]
        terms = [n.value(model.get("bias", 0.0), "Bias")]
        if model.get("trend_u", 0): terms.append(n.mul(model["trend_u"], n.sub(u, .5)))
        if model.get("trend_v", 0): terms.append(n.mul(model["trend_v"], n.sub(v, .5)))
        count = len(model["components"])
        for index, component in enumerate(model["components"]):
            kind = component["type"]
            before = set(tree.nodes)
            builder = BUILDERS[kind]
            term = builder(tree, n, u, v, component) if kind in APPROXIMATE_TYPES else builder(n, u, v, component)
            terms.append(term)
            created = list(set(tree.nodes) - before)
            if created:
                anchor = created[0]; anchor.name = f"PTK_{index:04d}_{kind}"; tag_component(anchor, index, kind)
            if progress: progress(index + 1, count)
        tree.links.new(n.balanced_add(terms), outputs.inputs["Value"])
        tree["ptk_approximate"] = bool(approximate)
        tree["ptk_estimated_nodes"] = estimate
        return tree, [f"{kind} component {path} uses a native Blender approximation" for path, kind in approximate]
    except Exception:
        bpy.data.node_groups.remove(tree)
        raise


def _fitted_coordinates(material_tree, group_node):
    tex = material_tree.nodes.new("ShaderNodeTexCoord"); tex.location = (-700, 0)
    sep = material_tree.nodes.new("ShaderNodeSeparateXYZ"); sep.location = (-500, 0)
    combine = material_tree.nodes.new("ShaderNodeCombineXYZ"); combine.location = (-300, 0)
    flip = material_tree.nodes.new("ShaderNodeMath"); flip.operation = "SUBTRACT"; flip.inputs[0].default_value = 1.0
    material_tree.links.new(tex.outputs["UV"], sep.inputs["Vector"])
    material_tree.links.new(sep.outputs["X"], combine.inputs["X"])
    material_tree.links.new(sep.outputs["Y"], flip.inputs[1])
    material_tree.links.new(flip.outputs[0], combine.inputs["Y"])
    material_tree.links.new(combine.outputs["Vector"], group_node.inputs["Fitted Coordinates"])


def create_material(group, name, source_path, document, warnings, route="ROUGHNESS"):
    material = bpy.data.materials.new(name); material.use_nodes = True
    tree = material.node_tree; tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial"); output.location = (650, 0)
    bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled"); bsdf.location = (350, 0)
    group_node = tree.nodes.new("ShaderNodeGroup"); group_node.node_tree = group; group_node.location = (-50, 0); group_node[NODE_TAG] = True
    _fitted_coordinates(tree, group_node)
    tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    value = group_node.outputs["Value"]
    if route == "ROUGHNESS":
        clamp = tree.nodes.new("ShaderNodeClamp"); tree.links.new(value, clamp.inputs["Value"]); tree.links.new(clamp.outputs["Result"], principled_input(bsdf, "roughness"))
    elif route == "BASE_COLOR": tree.links.new(value, principled_input(bsdf, "base_color"))
    elif route == "METALLIC": tree.links.new(value, principled_input(bsdf, "metallic"))
    elif route == "ALPHA": tree.links.new(value, principled_input(bsdf, "alpha")); material.surface_render_method = "DITHERED"
    elif route == "EMISSION": tree.links.new(value, principled_input(bsdf, "emission"))
    elif route == "BUMP":
        bump = tree.nodes.new("ShaderNodeBump"); tree.links.new(value, bump.inputs["Height"]); tree.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    material["ptk_source_path"] = str(Path(source_path).resolve())
    material["ptk_schema_version"] = 1
    material["ptk_metric_summary"] = metric_summary(document)
    material["ptk_import_warnings"] = json.dumps(warnings)
    material["ptk_generated"] = True
    return material


def assign_material(context, material):
    obj = context.active_object
    if obj is None or not hasattr(obj.data, "materials"):
        raise PTKImportError("The active object cannot receive materials")
    if obj.material_slots and obj.active_material_index < len(obj.material_slots):
        obj.material_slots[obj.active_material_index].material = material
    else: obj.data.materials.append(material)


def replace_group(material, group):
    if not material or not material.use_nodes:
        raise PTKImportError("The active material does not use nodes")
    node = next((x for x in material.node_tree.nodes if x.get(NODE_TAG)), None)
    if node is None: raise PTKImportError("The active material is not a PTK material")
    old = node.node_tree; node.node_tree = group
    if old and old.users == 0: bpy.data.node_groups.remove(old)
    return material
