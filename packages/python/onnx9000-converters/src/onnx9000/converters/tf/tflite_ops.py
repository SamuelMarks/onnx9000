"""Module providing tflite ops functionality."""

from typing import Callable

from onnx9000.converters.tf.builder import TFToONNXGraphBuilder
from onnx9000.converters.tf.parsers import TFNode


def _map_tflite_simple_binary(op_type: str) -> Callable:
    """Execute the  map tflite simple binary operation."""

    def _impl(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
        """Execute the  impl operation."""
        return builder.make_node(op_type, node.inputs, {}, node.name)

    return _impl


def _map_tflite_pool(op_type: str) -> Callable:
    """Execute the  map tflite pool operation."""

    def _impl(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
        """Execute the  impl operation."""
        return builder.make_node(f"Custom_TFLite{op_type}", node.inputs, node.attr, node.name)

    return _impl


def _map_tflite_conv2d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map tflite conv2d operation."""
    return builder.make_node("Custom_TFLiteConv2D", node.inputs, node.attr, node.name)


def _map_tflite_depthwise_conv2d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map tflite depthwise conv2d operation."""
    return builder.make_node("Custom_TFLiteDepthwiseConv2D", node.inputs, node.attr, node.name)


def _map_tflite_fully_connected(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map tflite fully connected operation."""
    return builder.make_node("Custom_TFLiteFullyConnected", node.inputs, node.attr, node.name)


def _map_tflite_reshape(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map tflite reshape operation."""
    return builder.make_node("Reshape", node.inputs, {}, node.name)


def _map_tflite_resize_bilinear(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map tflite resize bilinear operation."""
    return builder.make_node("Resize", node.inputs, {"mode": "linear"}, node.name)


def _map_tflite_concat(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map tflite concat operation."""
    axis = builder.extract_attr(node, "axis", 0)
    return builder.make_node("Concat", node.inputs, {"axis": axis}, node.name)


def _map_tflite_softmax(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map tflite softmax operation."""
    return builder.make_node("Softmax", node.inputs, {"axis": -1}, node.name)


def _map_tflite_logistic(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map tflite logistic operation."""
    return builder.make_node("Sigmoid", node.inputs, {}, node.name)


def _map_tflite_tanh(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map tflite tanh operation."""
    return builder.make_node("Tanh", node.inputs, {}, node.name)


def _map_tflite_relu(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map tflite relu operation."""
    return builder.make_node("Relu", node.inputs, {}, node.name)


def _map_tflite_relu6(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map tflite relu6 operation."""
    zero = builder.add_constant(f"{node.name}_zero", 0.0, 1, ())
    six = builder.add_constant(f"{node.name}_six", 6.0, 1, ())
    return builder.make_node("Clip", [node.inputs[0], zero, six], {}, node.name)


def _map_tflite_relu_n1_to_1(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map tflite relu n1 to 1 operation."""
    neg_one = builder.add_constant(f"{node.name}_neg1", -1.0, 1, ())
    pos_one = builder.add_constant(f"{node.name}_pos1", 1.0, 1, ())
    return builder.make_node("Clip", [node.inputs[0], neg_one, pos_one], {}, node.name)


def _map_tflite_dequantize(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map tflite dequantize operation."""
    return builder.make_node("DequantizeLinear", node.inputs, {}, node.name)


def _map_tflite_quantize(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map tflite quantize operation."""
    return builder.make_node("QuantizeLinear", node.inputs, {}, node.name)


def _map_tflite_embedding_lookup(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map tflite embedding lookup operation."""
    return builder.make_node("Gather", node.inputs, {}, node.name)


def _map_tflite_l2_normalization(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map tflite l2 normalization operation."""
    return builder.make_node("LpNormalization", node.inputs, {"p": 2}, node.name)


def _map_tflite_local_response_normalization(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map tflite local response normalization operation."""
    return builder.make_node("LRN", node.inputs, {}, node.name)


def _map_tflite_space_to_depth(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map tflite space to depth operation."""
    return builder.make_node("SpaceToDepth", node.inputs, {}, node.name)


def _map_tflite_depth_to_space(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map tflite depth to space operation."""
    return builder.make_node("DepthToSpace", node.inputs, {}, node.name)


def _map_tflite_floor(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map tflite floor operation."""
    return builder.make_node("Floor", node.inputs, {}, node.name)


def _map_tflite_custom_subgraph(op: str) -> Callable:
    """Execute the  map tflite custom subgraph operation."""

    def _impl(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
        """Execute the  impl operation."""
        return builder.make_node(f"Custom_TFLite{op}", node.inputs, node.attr, node.name)

    return _impl


TFLITE_OPS_MAPPING: dict[str, Callable[[TFToONNXGraphBuilder, TFNode], list[str]]] = {
    "ADD": _map_tflite_simple_binary("Add"),
    "MUL": _map_tflite_simple_binary("Mul"),
    "AVERAGE_POOL_2D": _map_tflite_pool("AveragePool2D"),
    "MAX_POOL_2D": _map_tflite_pool("MaxPool2D"),
    "L2_POOL_2D": _map_tflite_pool("L2Pool2D"),
    "CONCATENATION": _map_tflite_concat,
    "CONV_2D": _map_tflite_conv2d,
    "DEPTHWISE_CONV_2D": _map_tflite_depthwise_conv2d,
    "FULLY_CONNECTED": _map_tflite_fully_connected,
    "RESHAPE": _map_tflite_reshape,
    "RESIZE_BILINEAR": _map_tflite_resize_bilinear,
    "SOFTMAX": _map_tflite_softmax,
    "LOGISTIC": _map_tflite_logistic,
    "TANH": _map_tflite_tanh,
    "RELU": _map_tflite_relu,
    "RELU6": _map_tflite_relu6,
    "RELU_N1_TO_1": _map_tflite_relu_n1_to_1,
    "DEQUANTIZE": _map_tflite_dequantize,
    "QUANTIZE": _map_tflite_quantize,
    "EMBEDDING_LOOKUP": _map_tflite_embedding_lookup,
    "L2_NORMALIZATION": _map_tflite_l2_normalization,
    "LOCAL_RESPONSE_NORMALIZATION": _map_tflite_local_response_normalization,
    "SPACE_TO_DEPTH": _map_tflite_space_to_depth,
    "DEPTH_TO_SPACE": _map_tflite_depth_to_space,
    "FLOOR": _map_tflite_floor,
    "HASHTABLE_LOOKUP": _map_tflite_custom_subgraph("HashtableLookup"),
    "LSH_PROJECTION": _map_tflite_custom_subgraph("LshProjection"),
    "LSTM": _map_tflite_custom_subgraph("LSTM"),
    "RNN": _map_tflite_custom_subgraph("RNN"),
    "SVDF": _map_tflite_custom_subgraph("SVDF"),
    "CONCAT_EMBEDDINGS": _map_tflite_custom_subgraph("ConcatEmbeddings"),
    "SKIP_GRAM": _map_tflite_custom_subgraph("SkipGram"),
    "CALL": _map_tflite_custom_subgraph("Call"),
    "CUSTOM": _map_tflite_custom_subgraph("Custom"),
}
