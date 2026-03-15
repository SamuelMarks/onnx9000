"""Module docstring."""

from typing import Callable
from onnx9000.frontends.paddle.builder import PaddleToONNXGraphBuilder
from onnx9000.frontends.paddle.parsers import PaddleNode


def _map_allclose(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map allclose operation."""
    # allclose(a, b) = reduce_all(abs(a - b) <= atol + rtol * abs(b))
    inputs = node.inputs.get("Input", [])
    if not inputs:
        inputs = node.inputs.get("X", [])
    other = node.inputs.get("Other", [])
    if not other:
        other = node.inputs.get("Y", [])

    rtol = builder.extract_attr(node, "rtol", 1e-5)
    atol = builder.extract_attr(node, "atol", 1e-8)

    a = inputs[0]
    b = other[0]

    diff = builder.make_node("Sub", [a, b], {}, f"{node.name}_sub")[0]
    abs_diff = builder.make_node("Abs", [diff], {}, f"{node.name}_abs_diff")[0]

    abs_b = builder.make_node("Abs", [b], {}, f"{node.name}_abs_b")[0]
    rtol_const = builder.add_constant(f"{node.name}_rtol", [rtol], 1, [1])
    atol_const = builder.add_constant(f"{node.name}_atol", [atol], 1, [1])

    rtol_b = builder.make_node("Mul", [abs_b, rtol_const], {}, f"{node.name}_rtol_b")[0]
    tol = builder.make_node("Add", [rtol_b, atol_const], {}, f"{node.name}_tol")[0]

    less_eq = builder.make_node("LessOrEqual", [abs_diff, tol], {}, f"{node.name}_le")[
        0
    ]

    # Paddle allclose produces a scalar bool
    return builder.make_node("ReduceMin", [less_eq], {"keepdims": 0}, node.name)


"""Module docstring."""


def _map_logical_binary(op_type: str) -> Callable:
    """Executes the  map logical binary operation."""

    def _impl(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
        """Executes the  impl operation."""
        inputs = []
        if "X" in node.inputs:
            inputs.extend(node.inputs["X"])
        if "Y" in node.inputs:
            inputs.extend(node.inputs["Y"])
        return builder.make_node(op_type, inputs, {}, node.name)

    return _impl


def _map_logical_unary(op_type: str) -> Callable:
    """Executes the  map logical unary operation."""

    def _impl(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
        """Executes the  impl operation."""
        inputs = node.inputs.get("X", [])
        return builder.make_node(op_type, inputs, {}, node.name)

    return _impl


def _map_not_equal(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map not equal operation."""
    inputs = []
    if "X" in node.inputs:
        inputs.extend(node.inputs["X"])
    if "Y" in node.inputs:
        inputs.extend(node.inputs["Y"])
    eq_out = builder.make_node("Equal", inputs, {}, f"{node.name}_eq")[0]
    return builder.make_node("Not", [eq_out], {}, node.name)


def _map_reduce(op_type: str) -> Callable:
    """Executes the  map reduce operation."""

    def _impl(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
        """Executes the  impl operation."""
        inputs = node.inputs.get("X", [])
        keep_dim = builder.extract_attr(node, "keep_dim", False)
        attrs = {"keepdims": 1 if keep_dim else 0}

        dim = builder.extract_list_attr(node, "dim")
        if dim:
            axes_tensor = builder.add_constant(f"{node.name}_axes", dim, 7, (len(dim),))
            inputs.append(axes_tensor)

        return builder.make_node(op_type, inputs, attrs, node.name)

    return _impl


def _map_arg(op_type: str) -> Callable:
    """Executes the  map arg operation."""

    def _impl(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
        """Executes the  impl operation."""
        inputs = node.inputs.get("X", [])
        axis = builder.extract_attr(node, "axis", 0)
        keepdims = builder.extract_attr(node, "keepdims", False)
        return builder.make_node(
            op_type, inputs, {"axis": axis, "keepdims": 1 if keepdims else 0}, node.name
        )

    return _impl


def _map_cumsum(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map cumsum operation."""
    inputs = node.inputs.get("X", [])
    axis = builder.extract_attr(node, "axis", 0)
    axis_tensor = builder.add_constant(f"{node.name}_axis", [axis], 7, (1,))
    inputs.append(axis_tensor)

    exclusive = builder.extract_attr(node, "exclusive", False)
    reverse = builder.extract_attr(node, "reverse", False)

    attrs = {"exclusive": 1 if exclusive else 0, "reverse": 1 if reverse else 0}
    return builder.make_node("CumSum", inputs, attrs, node.name)


REDUCTION_OPS_MAPPING: dict[
    str, Callable[[PaddleToONNXGraphBuilder, PaddleNode], list[str]]
] = {
    "reduce_all": _map_reduce("ReduceMin"),
    "reduce_any": _map_reduce("ReduceMax"),
    "reduce_max": _map_reduce("ReduceMax"),
    "reduce_min": _map_reduce("ReduceMin"),
    "reduce_prod": _map_reduce("ReduceProd"),
    "reduce_sum": _map_reduce("ReduceSum"),
    "reduce_mean": _map_reduce("ReduceMean"),
    "arg_max": _map_arg("ArgMax"),
    "arg_min": _map_arg("ArgMin"),
    "logical_and": _map_logical_binary("And"),
    "logical_or": _map_logical_binary("Or"),
    "logical_xor": _map_logical_binary("Xor"),
    "logical_not": _map_logical_unary("Not"),
    "equal": _map_logical_binary("Equal"),
    "not_equal": _map_not_equal,
    "less_than": _map_logical_binary("Less"),
    "less_equal": _map_logical_binary("LessOrEqual"),
    "greater_than": _map_logical_binary("Greater"),
    "greater_equal": _map_logical_binary("GreaterOrEqual"),
    "allclose": _map_allclose,
    "cumsum": _map_cumsum,
}
