"""Module providing reduction ops functionality."""

from typing import Callable, Dict, List
from onnx9000.frontends.tf.builder import TFToONNXGraphBuilder
from onnx9000.frontends.tf.parsers import TFNode


def _map_reduce_op(onnx_op_type: str) -> Callable:
    """Executes the  map reduce op operation."""

    def _impl(builder: TFToONNXGraphBuilder, node: TFNode) -> List[str]:
        """Executes the  impl operation."""
        keepdims = builder.extract_attr(node, "keep_dims", False)
        return builder.make_node(
            onnx_op_type, node.inputs, {"keepdims": 1 if keepdims else 0}, node.name
        )

    return _impl


def _map_bincount(builder: TFToONNXGraphBuilder, node: TFNode) -> List[str]:
    """Executes the  map bincount operation."""
    return builder.make_node("Custom_Bincount", node.inputs, {}, node.name)


def _map_cumsum(builder: TFToONNXGraphBuilder, node: TFNode) -> List[str]:
    """Executes the  map cumsum operation."""
    exclusive = builder.extract_attr(node, "exclusive", False)
    reverse = builder.extract_attr(node, "reverse", False)
    return builder.make_node(
        "CumSum",
        node.inputs,
        {"exclusive": 1 if exclusive else 0, "reverse": 1 if reverse else 0},
        node.name,
    )


def _map_cumprod(builder: TFToONNXGraphBuilder, node: TFNode) -> List[str]:
    """Executes the  map cumprod operation."""
    # ONNX does not have CumProd natively, so map to Custom
    return builder.make_node("Custom_Cumprod", node.inputs, {}, node.name)


def _map_logical_binary(onnx_op_type: str) -> Callable:
    """Executes the  map logical binary operation."""

    def _impl(builder: TFToONNXGraphBuilder, node: TFNode) -> List[str]:
        """Executes the  impl operation."""
        return builder.make_node(onnx_op_type, node.inputs, {}, node.name)

    return _impl


def _map_not_equal(builder: TFToONNXGraphBuilder, node: TFNode) -> List[str]:
    """Executes the  map not equal operation."""
    eq = builder.make_node("Equal", node.inputs, {}, f"{node.name}_eq")[0]
    return builder.make_node("Not", [eq], {}, node.name)


REDUCTION_OPS_MAPPING: Dict[
    str, Callable[[TFToONNXGraphBuilder, TFNode], List[str]]
] = {
    "Sum": _map_reduce_op("ReduceSum"),
    "Mean": _map_reduce_op("ReduceMean"),
    "Prod": _map_reduce_op("ReduceProd"),
    "Max": _map_reduce_op("ReduceMax"),
    "Min": _map_reduce_op("ReduceMin"),
    "All": _map_reduce_op("ReduceMin"),  # Boolean context
    "Any": _map_reduce_op("ReduceMax"),  # Boolean context
    "ArgMax": _map_reduce_op("ArgMax"),
    "ArgMin": _map_reduce_op("ArgMin"),
    "Bincount": _map_bincount,
    "Cumsum": _map_cumsum,
    "Cumprod": _map_cumprod,
    "LogicalAnd": _map_logical_binary("And"),
    "LogicalOr": _map_logical_binary("Or"),
    "LogicalNot": _map_logical_binary("Not"),
    "LogicalXor": _map_logical_binary("Xor"),
    "Equal": _map_logical_binary("Equal"),
    "NotEqual": _map_not_equal,
    "Greater": _map_logical_binary("Greater"),
    "GreaterEqual": _map_logical_binary("GreaterOrEqual"),
    "Less": _map_logical_binary("Less"),
    "LessEqual": _map_logical_binary("LessOrEqual"),
}
