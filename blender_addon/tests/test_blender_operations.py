"""Static guard for Blender 4.5 ShaderNodeMath operation identifiers."""
import ast
from pathlib import Path

BLENDER_45_MATH_OPERATIONS = {
    "ADD", "SUBTRACT", "MULTIPLY", "DIVIDE", "MULTIPLY_ADD", "POWER",
    "LOGARITHM", "SQRT", "INVERSE_SQRT", "ABSOLUTE", "EXPONENT", "MINIMUM",
    "MAXIMUM", "LESS_THAN", "GREATER_THAN", "SIGN", "COMPARE", "SMOOTH_MIN",
    "SMOOTH_MAX", "ROUND", "FLOOR", "CEIL", "TRUNC", "FRACT", "MODULO",
    "FLOORED_MODULO", "WRAP", "SNAP", "PINGPONG", "SINE", "COSINE", "TANGENT",
    "ARCSINE", "ARCCOSINE", "ARCTANGENT", "ARCTAN2", "SINH", "COSH", "TANH",
    "RADIANS", "DEGREES",
}


def test_generated_math_operations_exist_in_blender_45():
    root = Path(__file__).parents[1]
    found = set()
    for path in root.rglob("*.py"):
        if "tests" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "math" and node.args
                    and isinstance(node.args[0], ast.Constant)):
                found.add(node.args[0].value)
    assert found <= BLENDER_45_MATH_OPERATIONS
