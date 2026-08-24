"""Material-properties UI."""
import bpy


class PTK_PT_material(bpy.types.Panel):
    bl_label = "Procedural Texture Kernel"; bl_idname = "PTK_PT_material"
    bl_space_type = "PROPERTIES"; bl_region_type = "WINDOW"; bl_context = "material"
    def draw(self, context):
        layout = self.layout
        layout.operator("ptk.import_material", icon="IMPORT")
        material = context.material
        if material and material.get("ptk_generated"):
            layout.operator("ptk.reimport_material", icon="FILE_REFRESH")
            layout.label(text=f"Schema: {material.get('ptk_schema_version', '?')}")
            layout.label(text=material.get("ptk_source_path", ""), icon="FILE")
