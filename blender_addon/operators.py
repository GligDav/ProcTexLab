"""Import and transactional reimport operators."""
from __future__ import annotations

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty, StringProperty
from bpy_extras.io_utils import ImportHelper

from .builder import assign_material, build_model_group, create_material, estimate_node_count, replace_group
from .schema import PTKImportError, load_document, validate_document


class PTK_OT_import_material(bpy.types.Operator, ImportHelper):
    bl_idname = "ptk.import_material"; bl_label = "Import PTK Fitter Result"; bl_options = {"REGISTER", "UNDO"}
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})
    allow_approximate: BoolProperty(name="Compact (Approximate)", default=False,
        description="Explicitly use native Blender textures for deterministic noise families")
    assign_to_active: BoolProperty(name="Assign to Active Object", default=True)
    node_limit: IntProperty(name="Hard Node Limit", default=100000, min=100, max=1000000)
    route: EnumProperty(name="Route Value To", items=[
        ("ROUGHNESS", "Roughness", "Clamped roughness"), ("BASE_COLOR", "Base Color", "Grayscale base color"),
        ("METALLIC", "Metallic", "Metallic"), ("ALPHA", "Alpha", "Alpha"),
        ("EMISSION", "Emission Strength", "Emission"), ("BUMP", "Bump Height", "Bump")], default="ROUGHNESS")

    def execute(self, context):
        group = material = None
        try:
            document = load_document(self.filepath); model = validate_document(document)
            estimate = estimate_node_count(model, self.allow_approximate)
            if estimate > self.node_limit: raise PTKImportError(f"Estimated {estimate:,} nodes exceed the hard limit")
            wm = context.window_manager; wm.progress_begin(0, max(1, len(model["components"])))
            group, warnings = build_model_group(model, "PTK_Model", self.allow_approximate, self.node_limit, lambda i, _: wm.progress_update(i))
            material = create_material(group, "PTK Material", self.filepath, document, warnings, self.route)
            if self.assign_to_active: assign_material(context, material)
            wm.progress_end()
        except Exception as exc:
            try: context.window_manager.progress_end()
            except Exception: pass
            if material is not None and material.users == 0: bpy.data.materials.remove(material)
            if group is not None and group.users == 0: bpy.data.node_groups.remove(group)
            self.report({"ERROR"}, str(exc)); return {"CANCELLED"}
        message = f"Imported {len(model['components'])} components ({estimate:,} estimated nodes)"
        self.report({"WARNING"} if warnings else {"INFO"}, message + (f"; {len(warnings)} approximations" if warnings else ""))
        return {"FINISHED"}


class PTK_OT_reimport_material(bpy.types.Operator):
    bl_idname = "ptk.reimport_material"; bl_label = "Reimport PTK Material"; bl_options = {"REGISTER", "UNDO"}
    allow_approximate: BoolProperty(name="Compact (Approximate)", default=False)
    node_limit: IntProperty(default=100000, min=100, max=1000000)
    def execute(self, context):
        material = context.active_object.active_material if context.active_object else None
        source = material.get("ptk_source_path") if material else None
        if not source: self.report({"ERROR"}, "Active material has no PTK source path"); return {"CANCELLED"}
        group = None
        try:
            document = load_document(source); model = validate_document(document)
            group, warnings = build_model_group(model, "PTK_Model", self.allow_approximate, self.node_limit)
            replace_group(material, group)
            material["ptk_import_warnings"] = __import__("json").dumps(warnings)
        except Exception as exc:
            if group is not None and group.users == 0: bpy.data.node_groups.remove(group)
            self.report({"ERROR"}, str(exc)); return {"CANCELLED"}
        self.report({"WARNING"} if warnings else {"INFO"}, "PTK material reimported")
        return {"FINISHED"}
