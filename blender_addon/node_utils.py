"""Concise helpers for generated scalar shader graphs."""
from __future__ import annotations


class Nodes:
    def __init__(self, tree):
        self.tree = tree
        self.nodes = tree.nodes
        self.links = tree.links
        self.serial = 0

    def value(self, value, label="Constant"):
        node = self.nodes.new("ShaderNodeValue")
        node.name = self._name(label); node.label = label
        node.outputs[0].default_value = float(value)
        return node.outputs[0]

    def math(self, operation, a, b=None, label=None):
        node = self.nodes.new("ShaderNodeMath")
        node.operation = operation
        node.name = self._name(label or operation); node.label = label or operation
        self._input(node.inputs[0], a)
        if b is not None: self._input(node.inputs[1], b)
        return node.outputs[0]

    def _input(self, target, value):
        if hasattr(value, "is_output"):
            self.links.new(value, target)
        else:
            target.default_value = float(value)

    def _name(self, label):
        self.serial += 1
        return f"PTK_{self.serial:05d}_{label.replace(' ', '_')}"

    def add(self, a, b): return self.math("ADD", a, b)
    def sub(self, a, b): return self.math("SUBTRACT", a, b)
    def mul(self, a, b): return self.math("MULTIPLY", a, b)
    def div(self, a, b): return self.math("DIVIDE", a, b)
    def power(self, a, b): return self.math("POWER", a, b)

    def balanced_add(self, values):
        values = list(values)
        if not values: return self.value(0.0)
        while len(values) > 1:
            values = [self.add(values[i], values[i + 1]) if i + 1 < len(values) else values[i]
                      for i in range(0, len(values), 2)]
        return values[0]


def tag_component(node, index, kind):
    node["ptk_component_index"] = index
    node["ptk_component_type"] = kind
