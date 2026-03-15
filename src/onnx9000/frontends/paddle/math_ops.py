"""Module docstring."""

from typing import Callable
from onnx9000.frontends.paddle.builder import PaddleToONNXGraphBuilder
from onnx9000.frontends.paddle.parsers import PaddleNode

import math


def _map_log2(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map log2 operation."""
    inputs = node.inputs.get("X", [])
    log_x = builder.make_node("Log", inputs, {}, f"{node.name}_log")[0]
    log_2_const = builder.add_constant(f"{node.name}_log2_const", math.log(2.0), 1, [])
    return builder.make_node("Div", [log_x, log_2_const], {}, node.name)


def _map_log10(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map log10 operation."""
    inputs = node.inputs.get("X", [])
    log_x = builder.make_node("Log", inputs, {}, f"{node.name}_log")[0]
    log_10_const = builder.add_constant(
        f"{node.name}_log10_const", math.log(10.0), 1, []
    )
    return builder.make_node("Div", [log_x, log_10_const], {}, node.name)


def _map_clip(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map clip operation."""
    inputs = node.inputs.get("X", [])
    min_val = builder.extract_attr(node, "min", -3.402823466e38)
    max_val = builder.extract_attr(node, "max", 3.402823466e38)

    # Clip in ONNX takes inputs: input, min (optional), max (optional)
    clip_inputs = list(inputs)
    min_const = builder.add_constant(f"{node.name}_min", float(min_val), 1, [])
    max_const = builder.add_constant(f"{node.name}_max", float(max_val), 1, [])
    clip_inputs.extend([min_const, max_const])

    return builder.make_node("Clip", clip_inputs, {}, node.name)


"""Module docstring."""


def _map_simple_binary(op_type: str) -> Callable:
    """Executes the  map simple binary operation."""

    def _impl(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
        """Executes the  impl operation."""
        inputs = []
        if "X" in node.inputs:
            inputs.extend(node.inputs["X"])
        if "Y" in node.inputs:
            inputs.extend(node.inputs["Y"])
        return builder.make_node(op_type, inputs, {}, node.name)

    return _impl


def _map_simple_unary(op_type: str) -> Callable:
    """Executes the  map simple unary operation."""

    def _impl(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
        """Executes the  impl operation."""
        inputs = node.inputs.get("X", [])
        return builder.make_node(op_type, inputs, {}, node.name)

    return _impl


def _map_floordiv(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map floordiv operation."""
    inputs = []
    if "X" in node.inputs:
        inputs.extend(node.inputs["X"])
    if "Y" in node.inputs:
        inputs.extend(node.inputs["Y"])
    div_out = builder.make_node("Div", inputs, {}, f"{node.name}_div")[0]
    return builder.make_node("Floor", [div_out], {}, node.name)


def _map_log1p(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map log1p operation."""
    inputs = node.inputs.get("X", [])
    one = builder.add_constant(f"{node.name}_one", 1.0, 1, ())
    add_out = builder.make_node("Add", inputs + [one], {}, f"{node.name}_add")[0]
    return builder.make_node("Log", [add_out], {}, node.name)


def _map_rsqrt(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map rsqrt operation."""
    inputs = node.inputs.get("X", [])
    sqrt_out = builder.make_node("Sqrt", inputs, {}, f"{node.name}_sqrt")[0]
    return builder.make_node("Reciprocal", [sqrt_out], {}, node.name)


def _map_square(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map square operation."""
    inputs = node.inputs.get("X", [])
    return builder.make_node("Mul", inputs * 2, {}, node.name)


def _map_isfinite(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map isfinite operation."""
    inputs = node.inputs.get("X", [])
    is_nan = builder.make_node("IsNaN", inputs, {}, f"{node.name}_isnan")[0]
    is_inf = builder.make_node("IsInf", inputs, {}, f"{node.name}_isinf")[0]
    or_out = builder.make_node("Or", [is_nan, is_inf], {}, f"{node.name}_or")[0]
    return builder.make_node("Not", [or_out], {}, node.name)


def _map_scale(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map scale operation."""
    inputs = node.inputs.get("X", [])
    scale = builder.extract_attr(node, "scale", 1.0)
    bias = builder.extract_attr(node, "bias", 0.0)

    scale_const = builder.add_constant(f"{node.name}_scale", scale, 1, ())
    bias_const = builder.add_constant(f"{node.name}_bias", bias, 1, ())

    mul_out = builder.make_node("Mul", inputs + [scale_const], {}, f"{node.name}_mul")[
        0
    ]
    return builder.make_node("Add", [mul_out, bias_const], {}, node.name)


def _map_sum(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map sum operation."""
    # Paddle sum usually means elementwise sum of a list of variables
    inputs = node.inputs.get("X", [])
    return builder.make_node("Sum", inputs, {}, node.name)


def _map_custom(op_name: str) -> Callable:
    """Executes the  map custom operation."""

    def _impl(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
        """Executes the  impl operation."""
        inputs = node.inputs.get("X", [])
        if "Y" in node.inputs:
            inputs.extend(node.inputs["Y"])
        return builder.make_node(op_name, inputs, {}, node.name)

    return _impl


MATH_OPS_MAPPING: dict[
    str, Callable[[PaddleToONNXGraphBuilder, PaddleNode], list[str]]
] = {
    "elementwise_add": _map_simple_binary("Add"),
    "elementwise_sub": _map_simple_binary("Sub"),
    "elementwise_mul": _map_simple_binary("Mul"),
    "elementwise_div": _map_simple_binary("Div"),
    "elementwise_mod": _map_simple_binary("Mod"),
    "elementwise_floordiv": _map_floordiv,
    "elementwise_max": _map_simple_binary("Max"),
    "elementwise_min": _map_simple_binary("Min"),
    "elementwise_pow": _map_simple_binary("Pow"),
    "abs": _map_simple_unary("Abs"),
    "exp": _map_simple_unary("Exp"),
    "log": _map_simple_unary("Log"),
    "log2": _map_log2,
    "log10": _map_log10,
    "log1p": _map_log1p,
    "pow": _map_simple_binary("Pow"),
    "square": _map_square,
    "sqrt": _map_simple_unary("Sqrt"),
    "rsqrt": _map_rsqrt,
    "reciprocal": _map_simple_unary("Reciprocal"),
    "ceil": _map_simple_unary("Ceil"),
    "floor": _map_simple_unary("Floor"),
    "round": _map_simple_unary("Round"),
    "sign": _map_simple_unary("Sign"),
    "sin": _map_simple_unary("Sin"),
    "cos": _map_simple_unary("Cos"),
    "tan": _map_simple_unary("Tan"),
    "asin": _map_simple_unary("Asin"),
    "acos": _map_simple_unary("Acos"),
    "atan": _map_simple_unary("Atan"),
    "sinh": _map_simple_unary("Sinh"),
    "cosh": _map_simple_unary("Cosh"),
    "tanh": _map_simple_unary("Tanh"),
    "asinh": _map_simple_unary("Asinh"),
    "acosh": _map_simple_unary("Acosh"),
    "atanh": _map_simple_unary("Atanh"),
    "erf": _map_simple_unary("Erf"),
    "isnan": _map_simple_unary("IsNaN"),
    "isinf": _map_simple_unary("IsInf"),
    "isfinite": _map_isfinite,
    "clip": _map_clip,
    "scale": _map_scale,
    "sum": _map_sum,
}
