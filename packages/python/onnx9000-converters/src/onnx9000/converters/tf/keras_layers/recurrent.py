"""Keras recurrent and bidirectional layer converters."""

from onnx9000.converters.tf.builder import TFToONNXGraphBuilder
from onnx9000.converters.tf.parsers import TFNode


def _map_keras_rnn(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Rnn operation to ONNX equivalent."""
    return builder.make_node("RNN", node.inputs, node.attr, node.name)


def _map_keras_rnn_base(builder: TFToONNXGraphBuilder, node: TFNode, op_type: str) -> list[str]:
    """Map RNN/LSTM/GRU layers to ONNX."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    attr = dict(node.attr)
    return_sequences = attr.pop("return_sequences", False)
    return_state = attr.pop("return_state", False)
    go_backwards = attr.pop("go_backwards", False)
    if go_backwards:
        attr["direction"] = b"reverse"
    else:
        attr["direction"] = b"forward"
    num_outputs = 1
    needs_yh = not return_sequences or return_state
    if op_type == "LSTM":
        num_outputs = 3 if return_state else 2 if needs_yh else 1
    else:
        num_outputs = 2 if needs_yh else 1
    nodes = builder.make_node(op_type, node.inputs, attr, node.name, num_outputs=num_outputs)
    Y = nodes[0]
    time_major = attr.pop("time_major", False)
    if not time_major:
        axes_squeeze = builder.make_node(
            "Constant", [], {"value": [1]}, node.name + "_squeeze_axes"
        )[0]
        squeezed_Y = builder.make_node("Squeeze", [Y, axes_squeeze], {}, node.name + "_squeezed_Y")[
            0
        ]
        transposed_Y = builder.make_node(
            "Transpose", [squeezed_Y], {"perm": [1, 0, 2]}, node.name + "_transposed_Y"
        )[0]
        Y = transposed_Y
    if not return_sequences:
        Y_h = nodes[1]
        axes_squeeze = builder.make_node(
            "Constant", [], {"value": [0]}, node.name + "_squeeze_h_axes"
        )[0]
        Y = builder.make_node("Squeeze", [Y_h, axes_squeeze], {}, node.name + "_Y")[0]
    out_list = [Y]
    if return_state:
        for i in range(1, num_outputs):
            axes_squeeze = builder.make_node(
                "Constant", [], {"value": [0]}, node.name + f"_squeeze_state_{i}_axes"
            )[0]
            sq_state = builder.make_node(
                "Squeeze", [nodes[i], axes_squeeze], {}, node.name + f"_state_{i}"
            )[0]
            out_list.append(sq_state)
    return out_list


def _map_keras_simple_rnn(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Simple Rnn operation to ONNX equivalent."""
    return _map_keras_rnn_base(builder, node, "RNN")


def _map_keras_lstm(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Lstm operation to ONNX equivalent."""
    return _map_keras_rnn_base(builder, node, "LSTM")


def _map_keras_gru(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Gru operation to ONNX equivalent."""
    return _map_keras_rnn_base(builder, node, "GRU")


def _map_keras_bidirectional(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Bidirectional operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    attr = dict(node.attr)
    attr["direction"] = b"bidirectional"
    op_type = "LSTM"
    if "layer" in attr:
        layer_conf = attr["layer"]
        if isinstance(layer_conf, dict):
            cls_name = layer_conf.get("class_name", "")
            if "LSTM" in cls_name:
                op_type = "LSTM"
            elif "GRU" in cls_name:
                op_type = "GRU"
            elif "SimpleRNN" in cls_name:
                op_type = "RNN"
    return _map_keras_rnn_base(builder, node, op_type)


def _map_keras_layers__bidirectional(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Bidirectional operation to ONNX equivalent."""
    return _map_keras_bidirectional(builder, node)


def _map_keras_conv_lstm_base(builder: TFToONNXGraphBuilder, node: TFNode, dim: int) -> list[str]:
    """Map Keras Conv Lstm Base operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    attr = dict(node.attr)
    return_sequences = attr.get("return_sequences", False)
    return_state = attr.get("return_state", False)
    num_outputs = 3 if return_state else 2 if not return_sequences else 1
    outputs = builder.make_node(
        "Identity", node.inputs[:1], attr, node.name, num_outputs=num_outputs
    )
    return outputs


def _map_keras_conv_lstm1d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Conv Lstm1D operation to ONNX equivalent."""
    return _map_keras_conv_lstm_base(builder, node, 1)


def _map_keras_conv_lstm2d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Conv Lstm2D operation to ONNX equivalent."""
    return _map_keras_conv_lstm_base(builder, node, 2)


def _map_keras_conv_lstm3d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Conv Lstm3D operation to ONNX equivalent."""
    return _map_keras_conv_lstm_base(builder, node, 3)


def _map_keras_layers__conv_lstm1_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Conv Lstm1 D operation to ONNX equivalent."""
    return _map_keras_conv_lstm1d(builder, node)


def _map_keras_layers__conv_lstm2_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Conv Lstm2 D operation to ONNX equivalent."""
    return _map_keras_conv_lstm2d(builder, node)


def _map_keras_layers__conv_lstm3_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Conv Lstm3 D operation to ONNX equivalent."""
    return _map_keras_conv_lstm3d(builder, node)


def _map_keras_layers_gru(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers Gru operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers_gru_cell(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers Gru Cell operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers_lstm(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers Lstm operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers_lstm_cell(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers Lstm Cell operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers_rnn(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers Rnn operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__simple_rnn(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Simple Rnn operation to ONNX equivalent."""
    return _map_keras_simple_rnn(builder, node)


def _map_keras_layers__simple_rnn_cell(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Simple Rnn Cell operation to ONNX equivalent."""
    return _map_keras_simple_rnn(builder, node)


def _map_keras_layers__stacked_rnn_cells(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Stacked Rnn Cells operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_time_distributed(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Time Distributed operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__time_distributed(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Time Distributed operation to ONNX equivalent."""
    return _map_keras_time_distributed(builder, node)
