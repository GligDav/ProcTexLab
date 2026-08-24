"""Translation of the fitter's small, topologically ordered scalar DAG."""


def shader_graph(n, u, v, component):
    # Imports are local to avoid a registry initialization cycle.
    from . import APPROXIMATE_TYPES, BUILDERS

    values = {}
    for item in component["graph"]["nodes"]:
        operation = item["operation"]
        args = [values[source] for source in item.get("inputs", ())]
        if operation == "component":
            embedded = item["component"]
            builder = BUILDERS[embedded["type"]]
            value = (builder(n.tree, n, u, v, embedded)
                     if embedded["type"] in APPROXIMATE_TYPES
                     else builder(n, u, v, embedded))
        elif operation == "constant":
            value = n.value(item.get("value", 0.0), "Graph constant")
        elif operation == "add": value = n.add(args[0], args[1])
        elif operation == "multiply": value = n.mul(args[0], args[1])
        elif operation == "one_minus": value = n.sub(1.0, args[0])
        elif operation == "smoothstep":
            edge0, edge1 = item.get("edge0", -0.08), item.get("edge1", 0.08)
            t = n.clamp(n.div(n.sub(args[0], edge0), edge1 - edge0))
            value = n.mul(n.mul(t, t), n.sub(3.0, n.mul(2.0, t)))
        else:  # mix(first, second, factor)
            value = n.add(n.mul(args[0], n.sub(1.0, args[2])), n.mul(args[1], args[2]))
        values[item["id"]] = value
    return n.mul(component["amplitude"], values[component["graph"]["output_node"]])
