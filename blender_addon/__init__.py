"""Blender add-on entry point for PTK fitter results."""
bl_info = {
    "name": "Procedural Texture Kernel Importer", "author": "ProcTexLab contributors",
    "version": (1, 0, 0), "blender": (4, 5, 0), "location": "Material Properties",
    "description": "Import PTK schema-v1 fitter JSON as an editable shader graph", "category": "Material",
}

def register():
    import bpy
    from .operators import PTK_OT_import_material, PTK_OT_reimport_material
    from .panels import PTK_PT_material
    global CLASSES
    CLASSES = (PTK_OT_import_material, PTK_OT_reimport_material, PTK_PT_material)
    for cls in CLASSES: bpy.utils.register_class(cls)


def unregister():
    import bpy
    for cls in reversed(globals().get("CLASSES", ())): bpy.utils.unregister_class(cls)
