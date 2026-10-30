"""Keras convolution layer converters."""

from onnx9000.converters.tf.builder import TFToONNXGraphBuilder
from onnx9000.converters.tf.parsers import TFNode


def _map_keras_conv_base(
    builder: TFToONNXGraphBuilder, node: TFNode, op_type: str = "Conv"
) -> list[str]:
    """Map attributes like dilations, groups, etc. for all Keras convolution layers."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    attr = dict(node.attr)
    if "dilation_rate" in attr:
        attr["dilations"] = attr.pop("dilation_rate")
    if "groups" in attr:
        attr["group"] = attr.pop("groups")
    if "padding" in attr:
        padding = attr.pop("padding").upper()
        if padding == "VALID":
            attr["auto_pad"] = b"VALID"
        elif padding == "SAME":
            attr["auto_pad"] = b"SAME_UPPER"
        elif padding == "CAUSAL":
            attr["auto_pad"] = b"VALID"
            kernel_size = attr.get("kernel_size", [1])[0]
            dilation_rate = attr.get("dilations", [1])[0]
            left_pad = dilation_rate * (kernel_size - 1)
            pads = [0, 0, left_pad, 0, 0, 0]
            pads_name = builder.make_node(
                "Constant", [], {"value": pads}, node.name + "_causal_pads"
            )[0]
            padded_input = builder.make_node(
                "Pad", [node.inputs[0], pads_name], {}, node.name + "_causal_pad"
            )[0]
            node.inputs[0] = padded_input
        else:
            attr["auto_pad"] = padding.encode("utf-8")
    return builder.make_node(op_type, node.inputs, attr, node.name)


def _map_keras_conv1d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Conv1D operation to ONNX equivalent."""
    return _map_keras_conv_base(builder, node)


def _map_keras_conv2d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Conv2D operation to ONNX equivalent."""
    return _map_keras_conv_base(builder, node)


def _map_keras_conv3d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Conv3D operation to ONNX equivalent."""
    return _map_keras_conv_base(builder, node)


def _map_keras_separable_conv1d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Separable Conv1D operation to ONNX equivalent."""
    return builder.make_node("Conv", node.inputs, node.attr, node.name)


def _map_keras_separable_conv2d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Separable Conv2D operation to ONNX equivalent."""
    return builder.make_node("Conv", node.inputs, node.attr, node.name)


def _map_keras_depthwise_conv1d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Depthwise Conv1D operation to ONNX equivalent."""
    return _map_keras_conv_base(builder, node)


def _map_keras_depthwise_conv2d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Depthwise Conv2D operation to ONNX equivalent."""
    return _map_keras_conv_base(builder, node)


def _map_keras_conv1d_transpose(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Conv1D Transpose operation to ONNX equivalent."""
    return _map_keras_conv_base(builder, node, "ConvTranspose")


def _map_keras_conv2d_transpose(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Conv2D Transpose operation to ONNX equivalent."""
    return _map_keras_conv_base(builder, node, "ConvTranspose")


def _map_keras_conv3d_transpose(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Conv3D Transpose operation to ONNX equivalent."""
    return _map_keras_conv_base(builder, node, "ConvTranspose")


def _map_keras_layers__conv1_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Conv1 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__convolution1_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Convolution1 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__conv1_d_transpose(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Conv1 D Transpose operation to ONNX equivalent."""
    return _map_keras_conv1d_transpose(builder, node)


def _map_keras_layers__convolution1_d_transpose(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Map Keras Layers  Convolution1 D Transpose operation to ONNX equivalent."""
    return _map_keras_conv1d_transpose(builder, node)


def _map_keras_layers__conv2_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Conv2 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__convolution2_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Convolution2 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__conv2_d_transpose(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Conv2 D Transpose operation to ONNX equivalent."""
    return _map_keras_conv2d_transpose(builder, node)


def _map_keras_layers__convolution2_d_transpose(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Map Keras Layers  Convolution2 D Transpose operation to ONNX equivalent."""
    return _map_keras_conv2d_transpose(builder, node)


def _map_keras_layers__conv3_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Conv3 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__convolution3_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Convolution3 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__conv3_d_transpose(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Conv3 D Transpose operation to ONNX equivalent."""
    return _map_keras_conv3d_transpose(builder, node)


def _map_keras_layers__convolution3_d_transpose(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Map Keras Layers  Convolution3 D Transpose operation to ONNX equivalent."""
    return _map_keras_conv3d_transpose(builder, node)


def _map_keras_layers__depthwise_conv1_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Depthwise Conv1 D operation to ONNX equivalent."""
    return _map_keras_depthwise_conv1d(builder, node)


def _map_keras_layers__depthwise_conv2_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Depthwise Conv2 D operation to ONNX equivalent."""
    return _map_keras_depthwise_conv2d(builder, node)


def _map_keras_layers__separable_conv1_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Separable Conv1 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__separable_convolution1_d(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Map Keras Layers  Separable Convolution1 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__separable_conv2_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Separable Conv2 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__separable_convolution2_d(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Map Keras Layers  Separable Convolution2 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)
