"""Small serializable scalar shader DAG used by experimental graph fitting."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .components import ProceduralComponent, component_from_dict

GRAPH_OPERATIONS = frozenset({
    "component", "constant", "add", "multiply", "smoothstep", "mix", "one_minus",
})


@dataclass
class ShaderNode:
    node_id: str
    operation: str
    inputs: tuple[str, ...] = ()
    component: ProceduralComponent | None = None
    value: float = 0.0
    edge0: float = -0.08
    edge1: float = 0.08

    def to_dict(self) -> dict:
        result = {"id": self.node_id, "operation": self.operation,
                  "inputs": list(self.inputs)}
        if self.operation == "component":
            result["component"] = self.component.to_dict()
        elif self.operation == "constant":
            result["value"] = float(self.value)
        elif self.operation == "smoothstep":
            result.update(edge0=float(self.edge0), edge1=float(self.edge1))
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "ShaderNode":
        operation = data.get("operation")
        component = (component_from_dict(data["component"])
                     if operation == "component" else None)
        return cls(str(data.get("id", "")), str(operation),
                   tuple(data.get("inputs", ())), component,
                   float(data.get("value", 0.0)), float(data.get("edge0", -.08)),
                   float(data.get("edge1", .08)))


@dataclass
class ShaderGraph:
    nodes: list[ShaderNode] = field(default_factory=list)
    output_node: str = ""

    def __post_init__(self):
        if not self.nodes:
            raise ValueError("shader graph must contain at least one node")
        if len(self.nodes) > 64:
            raise ValueError("shader graph exceeds the 64-node safety limit")
        seen = set()
        arity = {"component": 0, "constant": 0, "one_minus": 1,
                 "smoothstep": 1, "add": 2, "multiply": 2, "mix": 3}
        for node in self.nodes:
            if not node.node_id or node.node_id in seen:
                raise ValueError("shader graph node ids must be non-empty and unique")
            if node.operation not in GRAPH_OPERATIONS:
                raise ValueError(f"unsupported shader graph operation: {node.operation!r}")
            if len(node.inputs) != arity[node.operation]:
                raise ValueError(f"invalid input count for shader node {node.node_id!r}")
            if any(source not in seen for source in node.inputs):
                raise ValueError("shader graph inputs must reference earlier nodes")
            if node.operation == "component" and node.component is None:
                raise ValueError("component shader node requires a component")
            if node.operation == "smoothstep" and node.edge1 <= node.edge0:
                raise ValueError("smoothstep edge1 must exceed edge0")
            seen.add(node.node_id)
        if self.output_node not in seen:
            raise ValueError("shader graph output node does not exist")

    def evaluate(self, u, v):
        values = {}
        for node in self.nodes:
            args = [values[source] for source in node.inputs]
            if node.operation == "component":
                value = node.component.evaluate(u, v)
            elif node.operation == "constant":
                value = np.full(np.broadcast_shapes(np.shape(u), np.shape(v)), node.value)
            elif node.operation == "add":
                value = args[0] + args[1]
            elif node.operation == "multiply":
                value = args[0] * args[1]
            elif node.operation == "one_minus":
                value = 1.0 - args[0]
            elif node.operation == "smoothstep":
                t = np.clip((args[0] - node.edge0) / (node.edge1-node.edge0), 0, 1)
                value = t*t*(3-2*t)
            else:  # mix: inputs are first, second, factor
                value = args[0] * (1.0-args[2]) + args[1] * args[2]
            values[node.node_id] = value
        return values[self.output_node]

    def to_dict(self) -> dict:
        return {"nodes": [node.to_dict() for node in self.nodes],
                "output_node": self.output_node}

    @classmethod
    def from_dict(cls, data: dict) -> "ShaderGraph":
        return cls([ShaderNode.from_dict(item) for item in data.get("nodes", ())],
                   str(data.get("output_node", "")))


@dataclass
class ShaderGraphComponent(ProceduralComponent):
    """One graph output embedded as an atom in the sparse outer model."""
    graph: ShaderGraph | None = None
    type_name = "shader_graph"

    def basis(self, u, v):
        if self.graph is None:
            raise ValueError("shader graph component requires a graph")
        return self.graph.evaluate(u, v)

    def to_dict(self) -> dict:
        if self.graph is None:
            raise ValueError("shader graph component requires a graph")
        return {"type": self.type_name, "amplitude": float(self.amplitude),
                "graph": self.graph.to_dict()}

    @classmethod
    def from_dict(cls, data: dict) -> "ShaderGraphComponent":
        return cls(float(data.get("amplitude", 1.0)), ShaderGraph.from_dict(data["graph"]))
