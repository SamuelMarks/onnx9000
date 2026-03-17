"""Module providing control flow ops functionality."""

from typing import Callable

from onnx9000.converters.tf.builder import TFToONNXGraphBuilder
from onnx9000.converters.tf.parsers import TFNode


def _map_noop(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map noop operation."""
    return builder.make_node(f"Custom_TF{node.op}", node.inputs, {}, node.name)


def _map_if(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map if operation."""
    then_branch = builder.extract_attr(node, "then_branch")
    else_branch = builder.extract_attr(node, "else_branch")
    return builder.make_node(
        "If", node.inputs, {"then_branch": then_branch, "else_branch": else_branch}, node.name
    )


def _map_while(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map while operation."""
    body = builder.extract_attr(node, "body")
    cond = builder.extract_attr(node, "cond")
    return builder.make_node("Loop", node.inputs, {"body": body, "cond": cond}, node.name)


def _map_tensor_array(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map tensor array operation."""
    dtype = builder.extract_attr(node, "dtype", 1)
    handle = builder.make_node("SequenceEmpty", [], {"dtype": dtype}, f"{node.name}_seq")[0]
    flow = builder.add_constant(f"{node.name}_flow", 0.0, 1, ())
    return [handle, flow]


def _map_tensor_array_read(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map tensor array read operation."""
    return builder.make_node("SequenceAt", [node.inputs[0], node.inputs[1]], {}, node.name)


def _map_tensor_array_write(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map tensor array write operation."""
    return builder.make_node(
        "SequenceInsert", [node.inputs[0], node.inputs[2], node.inputs[1]], {}, node.name
    )


def _map_tensor_array_size(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map tensor array size operation."""
    return builder.make_node("SequenceLength", [node.inputs[0]], {}, node.name)


def _map_tensor_array_gather(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map tensor array gather operation."""
    return builder.make_node(
        "ConcatFromSequence", [node.inputs[0]], {"axis": 0, "new_axis": 1}, node.name
    )


def _map_tensor_array_scatter(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map tensor array scatter operation."""
    return builder.make_node(
        "SplitToSequence", [node.inputs[2]], {"axis": 0, "keepdims": 0}, node.name
    )


def _map_lstm_cell(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map lstm cell operation."""
    return builder.make_node("LSTM", node.inputs, {}, node.name)


def _map_gru_cell(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map gru cell operation."""
    return builder.make_node("GRU", node.inputs, {}, node.name)


CONTROL_FLOW_OPS_MAPPING: dict[str, Callable[[TFToONNXGraphBuilder, TFNode], list[str]]] = {
    "Enter": _map_noop,
    "Exit": _map_noop,
    "Merge": _map_noop,
    "Switch": _map_noop,
    "NextIteration": _map_noop,
    "LoopCond": _map_noop,
    "StatelessIf": _map_if,
    "If": _map_if,
    "StatelessWhile": _map_while,
    "While": _map_while,
    "TensorArrayV3": _map_tensor_array,
    "TensorArrayReadV3": _map_tensor_array_read,
    "TensorArrayWriteV3": _map_tensor_array_write,
    "TensorArraySizeV3": _map_tensor_array_size,
    "TensorArrayGatherV3": _map_tensor_array_gather,
    "TensorArrayScatterV3": _map_tensor_array_scatter,
    "BasicLSTMCell": _map_lstm_cell,
    "LSTMBlockCell": _map_lstm_cell,
    "BlockLSTM": _map_lstm_cell,
    "GRUBlockCell": _map_gru_cell,
}
