"""Keras merging layer converters."""

from collections.abc import Callable

from onnx9000.converters.tf.builder import TFToONNXGraphBuilder
from onnx9000.converters.tf.parsers import TFNode


def _map_keras_zero_padding(dim: str) -> Callable:
    """Map Keras Zero Padding operation to ONNX equivalent."""

    def _impl(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
        """Execute the impl operation."""
        if not node.inputs:
            return builder.make_node("Constant", [], node.attr, node.name)
        attr = dict(node.attr)
        padding = attr.pop("padding", None)
        pads_begin = [0, 0]
        pads_end = [0, 0]
        if padding is None or isinstance(padding, str):
            if dim == "1D":
                padding = (1, 1)
            elif dim == "2D":
                padding = ((1, 1), (1, 1))
            else:
                padding = ((1, 1), (1, 1), (1, 1))
        if isinstance(padding, int):
            if dim == "1D":
                padding = (padding, padding)
            elif dim == "2D":
                padding = ((padding, padding), (padding, padding))
            else:
                padding = ((padding, padding), (padding, padding), (padding, padding))
        if dim == "1D":
            pads_begin.append(padding[0])
            pads_end.append(padding[1])
        elif dim == "2D":
            if isinstance(padding[0], int):
                padding = ((padding[0], padding[0]), (padding[1], padding[1]))
            pads_begin.extend([padding[0][0], padding[1][0]])
            pads_end.extend([padding[0][1], padding[1][1]])
        else:
            if isinstance(padding[0], int):
                padding = (
                    (padding[0], padding[0]),
                    (padding[1], padding[1]),
                    (padding[2], padding[2]),
                )
            pads_begin.extend([padding[0][0], padding[1][0], padding[2][0]])
            pads_end.extend([padding[0][1], padding[1][1], padding[2][1]])
        pads = pads_begin + pads_end
        pads_name = builder.make_node("Constant", [], {"value": pads}, node.name + "_pads")[0]
        return builder.make_node("Pad", [node.inputs[0], pads_name], attr, node.name)

    return _impl


def _map_keras_concatenate(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Concatenate operation to ONNX equivalent."""
    return builder.make_node("Concat", node.inputs, node.attr, node.name)


def _map_keras_average(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Average operation to ONNX equivalent."""
    return builder.make_node("Mean", node.inputs, node.attr, node.name)


def _map_keras_maximum(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Maximum operation to ONNX equivalent."""
    return builder.make_node("Max", node.inputs, node.attr, node.name)


def _map_keras_minimum(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Minimum operation to ONNX equivalent."""
    return builder.make_node("Min", node.inputs, node.attr, node.name)


def _map_keras_add(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Add operation to ONNX equivalent."""
    return builder.make_node("Sum", node.inputs, node.attr, node.name)


def _map_keras_subtract(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Subtract operation to ONNX equivalent."""
    return builder.make_node("Sub", node.inputs, node.attr, node.name)


def _map_keras_multiply(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Multiply operation to ONNX equivalent."""
    return builder.make_node("Mul", node.inputs, node.attr, node.name)


def _map_keras_dot(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Dot operation to ONNX equivalent."""
    return builder.make_node("MatMul", node.inputs, node.attr, node.name)


def _map_keras_layers__add(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Add operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers_add(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers Add operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__average(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Average operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers_average(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers Average operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__concatenate(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Concatenate operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers_concatenate(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers Concatenate operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__dot(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Dot operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers_dot(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers Dot operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__maximum(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Maximum operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers_maximum(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers Maximum operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__minimum(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Minimum operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers_minimum(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers Minimum operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__multiply(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Multiply operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers_multiply(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers Multiply operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__subtract(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Subtract operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers_subtract(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers Subtract operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__zero_padding1_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Zero Padding1 D operation to ONNX equivalent."""
    return _map_keras_zero_padding("1D")(builder, node)


def _map_keras_layers__zero_padding2_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Zero Padding2 D operation to ONNX equivalent."""
    return _map_keras_zero_padding("2D")(builder, node)


def _map_keras_layers__zero_padding3_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Zero Padding3 D operation to ONNX equivalent."""
    return _map_keras_zero_padding("3D")(builder, node)
