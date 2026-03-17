"""Module providing math ops functionality."""

from typing import Callable

from onnx9000.converters.tf.builder import TFToONNXGraphBuilder
from onnx9000.converters.tf.parsers import TFNode


def _map_simple_binary(op_type: str) -> Callable:
    """Executes the  map simple binary operation."""

    def _impl(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
        """Executes the  impl operation."""
        return builder.make_node(op_type, node.inputs, {}, node.name)

    return _impl


def _map_simple_unary(op_type: str) -> Callable:
    """Executes the  map simple unary operation."""

    def _impl(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
        """Executes the  impl operation."""
        return builder.make_node(op_type, node.inputs, {}, node.name)

    return _impl


def _map_floor_div(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map floor div operation."""
    div_out = builder.make_node("Div", node.inputs, {}, f"{node.name}_div")[0]
    return builder.make_node("Floor", [div_out], {}, node.name)


def _map_floor_mod(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map floor mod operation."""
    return builder.make_node("Mod", node.inputs, {"fmod": 0}, node.name)


def _map_square(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map square operation."""
    return builder.make_node("Mul", [node.inputs[0], node.inputs[0]], {}, node.name)


def _map_rsqrt(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map rsqrt operation."""
    sqrt_out = builder.make_node("Sqrt", node.inputs, {}, f"{node.name}_sqrt")[0]
    return builder.make_node("Reciprocal", [sqrt_out], {}, node.name)


def _map_expm1(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map expm1 operation."""
    exp_out = builder.make_node("Exp", node.inputs, {}, f"{node.name}_exp")[0]
    one = builder.add_constant(f"{node.name}_one", 1.0, 1, ())
    return builder.make_node("Sub", [exp_out, one], {}, node.name)


def _map_log1p(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map log1p operation."""
    one = builder.add_constant(f"{node.name}_one", 1.0, 1, ())
    add_out = builder.make_node("Add", [node.inputs[0], one], {}, f"{node.name}_add")[0]
    return builder.make_node("Log", [add_out], {}, node.name)


def _map_atan2(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map atan2 operation."""
    return builder.make_node("Custom_Atan2", node.inputs, {}, node.name)


def _map_isfinite(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map isfinite operation."""
    is_nan = builder.make_node("IsNaN", node.inputs, {}, f"{node.name}_isnan")[0]
    is_inf = builder.make_node("IsInf", node.inputs, {}, f"{node.name}_isinf")[0]
    or_out = builder.make_node("Or", [is_nan, is_inf], {}, f"{node.name}_or")[0]
    return builder.make_node("Not", [or_out], {}, node.name)


def _map_complex_abs(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map complex abs operation."""
    return builder.make_node("Custom_ComplexAbs", node.inputs, {}, node.name)


def _map_angle(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map angle operation."""
    return builder.make_node("Custom_Angle", node.inputs, {}, node.name)


MATH_OPS_MAPPING: dict[str, Callable[[TFToONNXGraphBuilder, TFNode], list[str]]] = {
    "Add": _map_simple_binary("Add"),
    "AddV2": _map_simple_binary("Add"),
    "Sub": _map_simple_binary("Sub"),
    "Mul": _map_simple_binary("Mul"),
    "Div": _map_simple_binary("Div"),
    "RealDiv": _map_simple_binary("Div"),
    "TruncateDiv": _map_simple_binary("Div"),
    "FloorDiv": _map_floor_div,
    "Mod": _map_simple_binary("Mod"),
    "FloorMod": _map_floor_mod,
    "Abs": _map_simple_unary("Abs"),
    "Neg": _map_simple_unary("Neg"),
    "Sign": _map_simple_unary("Sign"),
    "Reciprocal": _map_simple_unary("Reciprocal"),
    "Square": _map_square,
    "Sqrt": _map_simple_unary("Sqrt"),
    "Rsqrt": _map_rsqrt,
    "Exp": _map_simple_unary("Exp"),
    "Expm1": _map_expm1,
    "Log": _map_simple_unary("Log"),
    "Log1p": _map_log1p,
    "Ceil": _map_simple_unary("Ceil"),
    "Floor": _map_simple_unary("Floor"),
    "Round": _map_simple_unary("Round"),
    "Maximum": _map_simple_binary("Max"),
    "Minimum": _map_simple_binary("Min"),
    "Sin": _map_simple_unary("Sin"),
    "Cos": _map_simple_unary("Cos"),
    "Tan": _map_simple_unary("Tan"),
    "Asin": _map_simple_unary("Asin"),
    "Acos": _map_simple_unary("Acos"),
    "Atan": _map_simple_unary("Atan"),
    "Atan2": _map_atan2,
    "Sinh": _map_simple_unary("Sinh"),
    "Cosh": _map_simple_unary("Cosh"),
    "Tanh": _map_simple_unary("Tanh"),
    "Asinh": _map_simple_unary("Asinh"),
    "Acosh": _map_simple_unary("Acosh"),
    "Atanh": _map_simple_unary("Atanh"),
    "Erf": _map_simple_unary("Erf"),
    "IsNan": _map_simple_unary("IsNaN"),
    "IsInf": _map_simple_unary("IsInf"),
    "IsFinite": _map_isfinite,
    "Pow": _map_simple_binary("Pow"),
    "ComplexAbs": _map_complex_abs,
    "Angle": _map_angle,
}
