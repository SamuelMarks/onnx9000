"""Module docstring."""

from typing import Callable

from onnx9000.frontends.paddle.builder import PaddleToONNXGraphBuilder
from onnx9000.frontends.paddle.parsers import PaddleNode


def _map_conditional_block(
    builder: PaddleToONNXGraphBuilder, node: PaddleNode
) -> list[str]:
    """Executes the  map conditional block operation."""
    return builder.make_node("If", node.inputs.get("Cond", []), node.attrs, node.name)


def _map_while(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map while operation."""
    # ONNX Loop usually maps roughly to Paddle's while, with body attributes
    return builder.make_node("Loop", node.inputs.get("X", []), node.attrs, node.name)


def _map_rnn(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map rnn operation."""
    return builder.make_node("RNN", node.inputs.get("Input", []), node.attrs, node.name)


def _map_lstm(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map lstm operation."""
    return builder.make_node(
        "LSTM", node.inputs.get("Input", []), node.attrs, node.name
    )


def _map_gru(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
    """Executes the  map gru operation."""
    return builder.make_node("GRU", node.inputs.get("Input", []), node.attrs, node.name)


def _map_tensor_array(op_type: str) -> Callable:
    """Executes the  map tensor array operation."""

    def _impl(builder: PaddleToONNXGraphBuilder, node: PaddleNode) -> list[str]:
        """Executes the  impl operation."""
        return builder.make_node(
            op_type,
            node.inputs.get("X", []) + node.inputs.get("Value", []),
            node.attrs,
            node.name,
        )

    return _impl


CONTROL_FLOW_OPS_MAPPING: dict[
    str, Callable[[PaddleToONNXGraphBuilder, PaddleNode], list[str]]
] = {
    "conditional_block": _map_conditional_block,
    "while": _map_while,
    "rnn": _map_rnn,
    "lstm": _map_lstm,
    "gru": _map_gru,
    "tensor_array_to_tensor": _map_tensor_array("ConcatFromSequence"),
    "lod_array_length": _map_tensor_array("SequenceLength"),
    "write_to_array": _map_tensor_array("SequenceInsert"),
    "read_from_array": _map_tensor_array("SequenceAt"),
}
