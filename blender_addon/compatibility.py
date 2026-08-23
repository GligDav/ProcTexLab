"""Small Blender 4.x compatibility boundary."""

def add_group_socket(tree, name, in_out, socket_type):
    return tree.interface.new_socket(name=name, in_out=in_out, socket_type=socket_type)


def socket(collection, name, fallback=None):
    found = collection.get(name)
    if found is not None:
        return found
    if fallback is not None:
        return collection[fallback]
    raise KeyError(f"socket {name!r} is unavailable")


def principled_input(node, purpose):
    alternatives = {
        "base_color": ("Base Color",), "roughness": ("Roughness",),
        "metallic": ("Metallic",), "alpha": ("Alpha",),
        "emission": ("Emission Strength", "Emission"),
    }
    for name in alternatives[purpose]:
        result = node.inputs.get(name)
        if result is not None:
            return result
    raise KeyError(f"Principled BSDF has no compatible {purpose} input")
