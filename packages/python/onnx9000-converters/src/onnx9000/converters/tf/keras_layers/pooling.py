"""Keras pooling layer converters."""

from onnx9000.converters.tf.builder import TFToONNXGraphBuilder
from onnx9000.converters.tf.parsers import TFNode


def _map_keras_pool_base(builder: TFToONNXGraphBuilder, node: TFNode, op_type: str) -> list[str]:
    """Map attributes like padding, strides, etc. for all Keras pooling layers."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    attr = dict(node.attr)
    if "padding" in attr:
        padding = attr.pop("padding").upper()
        if padding == "VALID":
            attr["auto_pad"] = b"VALID"
        elif padding == "SAME":
            attr["auto_pad"] = b"SAME_UPPER"
        else:
            attr["auto_pad"] = padding.encode("utf-8")
    return builder.make_node(op_type, node.inputs, attr, node.name)


def _map_keras_max_pooling_1d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Max Pooling 1D operation to ONNX equivalent."""
    return _map_keras_pool_base(builder, node, "MaxPool")


def _map_keras_max_pooling_2d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Max Pooling 2D operation to ONNX equivalent."""
    return _map_keras_pool_base(builder, node, "MaxPool")


def _map_keras_max_pooling_3d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Max Pooling 3D operation to ONNX equivalent."""
    return _map_keras_pool_base(builder, node, "MaxPool")


def _map_keras_avg_pooling_1d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Avg Pooling 1D operation to ONNX equivalent."""
    return _map_keras_pool_base(builder, node, "AveragePool")


def _map_keras_avg_pooling_2d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Avg Pooling 2D operation to ONNX equivalent."""
    return _map_keras_pool_base(builder, node, "AveragePool")


def _map_keras_avg_pooling_3d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Avg Pooling 3D operation to ONNX equivalent."""
    return _map_keras_pool_base(builder, node, "AveragePool")


def _map_keras_global_max_pooling_1d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Global Max Pooling 1D operation to ONNX equivalent."""
    return builder.make_node("GlobalMaxPool", node.inputs, node.attr, node.name)


def _map_keras_global_max_pooling_2d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Global Max Pooling 2D operation to ONNX equivalent."""
    return builder.make_node("GlobalMaxPool", node.inputs, node.attr, node.name)


def _map_keras_global_max_pooling_3d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Global Max Pooling 3D operation to ONNX equivalent."""
    return builder.make_node("GlobalMaxPool", node.inputs, node.attr, node.name)


def _map_keras_global_avg_pooling_1d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Global Avg Pooling 1D operation to ONNX equivalent."""
    return builder.make_node("GlobalAveragePool", node.inputs, node.attr, node.name)


def _map_keras_global_avg_pooling_2d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Global Avg Pooling 2D operation to ONNX equivalent."""
    return builder.make_node("GlobalAveragePool", node.inputs, node.attr, node.name)


def _map_keras_global_avg_pooling_3d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Global Avg Pooling 3D operation to ONNX equivalent."""
    return builder.make_node("GlobalAveragePool", node.inputs, node.attr, node.name)


def _map_keras_layers__adaptive_average_pooling1_d(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Map Keras Layers  Adaptive Average Pooling1 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__adaptive_average_pooling2_d(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Map Keras Layers  Adaptive Average Pooling2 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__adaptive_average_pooling3_d(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Map Keras Layers  Adaptive Average Pooling3 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__adaptive_max_pooling1_d(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Map Keras Layers  Adaptive Max Pooling1 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__adaptive_max_pooling2_d(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Map Keras Layers  Adaptive Max Pooling2 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__adaptive_max_pooling3_d(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Map Keras Layers  Adaptive Max Pooling3 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__average_pooling1_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Average Pooling1 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__avg_pool1_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Avg Pool1 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__average_pooling2_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Average Pooling2 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__avg_pool2_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Avg Pool2 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__average_pooling3_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Average Pooling3 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__avg_pool3_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Avg Pool3 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__global_average_pooling1_d(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Map Keras Layers  Global Average Pooling1 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__global_avg_pool1_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Global Avg Pool1 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__global_average_pooling2_d(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Map Keras Layers  Global Average Pooling2 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__global_avg_pool2_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Global Avg Pool2 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__global_average_pooling3_d(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Map Keras Layers  Global Average Pooling3 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__global_avg_pool3_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Global Avg Pool3 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__global_max_pool1_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Global Max Pool1 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__global_max_pooling1_d(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Map Keras Layers  Global Max Pooling1 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__global_max_pool2_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Global Max Pool2 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__global_max_pooling2_d(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Map Keras Layers  Global Max Pooling2 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__global_max_pool3_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Global Max Pool3 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__global_max_pooling3_d(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Map Keras Layers  Global Max Pooling3 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__max_pool1_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Max Pool1 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__max_pooling1_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Max Pooling1 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__max_pool2_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Max Pool2 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__max_pooling2_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Max Pooling2 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__max_pool3_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Max Pool3 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__max_pooling3_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Max Pooling3 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)
