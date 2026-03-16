"""Module providing nn ops functionality."""

from typing import Callable
from onnx9000.frontend.tf.builder import TFToONNXGraphBuilder
from onnx9000.frontend.tf.parsers import TFNode


def _map_simple_binary(op_type: str) -> Callable:
    """Executes the  map simple binary operation."""

    def _impl(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
        """Executes the  impl operation."""
        return builder.make_node(op_type, node.inputs, {}, node.name)

    return _impl


def _map_simple_unary(op_type: str) -> Callable:
    """Executes the  map simple unary operation."""

    def _impl(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
        """Executes the  impl operation."""
        return builder.make_node(op_type, node.inputs, {}, node.name)

    return _impl


def _map_relu6(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map relu6 operation."""
    zero = builder.add_constant(f"{node.name}_zero", 0.0, 1, ())
    six = builder.add_constant(f"{node.name}_six", 6.0, 1, ())
    return builder.make_node("Clip", [node.inputs[0], zero, six], {}, node.name)


def _map_leaky_relu(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map leaky relu operation."""
    alpha = builder.extract_attr(node, "alpha", 0.2)
    return builder.make_node("LeakyRelu", node.inputs, {"alpha": alpha}, node.name)


def _map_elu(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map elu operation."""
    alpha = builder.extract_attr(node, "alpha", 1.0)
    return builder.make_node("Elu", node.inputs, {"alpha": alpha}, node.name)


def _map_selu(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map selu operation."""
    alpha = builder.extract_attr(node, "alpha", 1.67326)
    gamma = builder.extract_attr(node, "gamma", 1.0507)
    return builder.make_node("Selu", node.inputs, {"alpha": alpha, "gamma": gamma}, node.name)


def _map_softplus(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map softplus operation."""
    return builder.make_node("Softplus", node.inputs, {}, node.name)


def _map_softsign(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map softsign operation."""
    return builder.make_node("Softsign", node.inputs, {}, node.name)


def _map_softmax(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map softmax operation."""
    axis = builder.extract_attr(node, "axis", -1)
    return builder.make_node("Softmax", node.inputs, {"axis": axis}, node.name)


def _map_log_softmax(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map log softmax operation."""
    axis = builder.extract_attr(node, "axis", -1)
    return builder.make_node("LogSoftmax", node.inputs, {"axis": axis}, node.name)


def _map_conv2d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map conv2d operation."""
    data_format = builder.extract_attr(node, "data_format", b"NHWC")
    if isinstance(data_format, bytes):
        data_format = data_format.decode("utf-8")
    strides = builder.extract_attr(node, "strides", [1, 1, 1, 1])
    dilations = builder.extract_attr(node, "dilations", [1, 1, 1, 1])
    input_t = node.inputs[0] if len(node.inputs) > 0 else ""
    weight_t = node.inputs[1] if len(node.inputs) > 1 else ""
    if data_format == "NHWC":
        input_t = builder.convert_nhwc_to_nchw(input_t)
        if len(strides) == 4:
            strides = [strides[1], strides[2]]
        if len(dilations) == 4:
            dilations = [dilations[1], dilations[2]]
    else:
        if len(strides) == 4:
            strides = [strides[2], strides[3]]
        if len(dilations) == 4:
            dilations = [dilations[2], dilations[3]]
    w_trans = builder.make_node(
        "Transpose", [weight_t], {"perm": [3, 2, 0, 1]}, f"{node.name}_w_trans"
    )[0]
    conv_out = builder.make_node(
        "Conv",
        [input_t, w_trans],
        {"strides": strides, "dilations": dilations},
        f"{node.name}_conv",
    )[0]
    if data_format == "NHWC":
        conv_out = builder.make_node(
            "Transpose", [conv_out], {"perm": [0, 2, 3, 1]}, f"{node.name}_nchw_to_nhwc"
        )[0]
    return [conv_out]


def _map_conv3d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map conv3d operation."""
    return builder.make_node("Conv", node.inputs, {}, node.name)


def _map_depthwise_conv2d_native(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map depthwise conv2d native operation."""
    return builder.make_node("Conv", node.inputs, {}, node.name)


def _map_conv2d_backprop_input(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map conv2d backprop input operation."""
    return builder.make_node("ConvTranspose", node.inputs, {}, node.name)


def _map_conv3d_backprop_input_v2(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map conv3d backprop input v2 operation."""
    return builder.make_node("ConvTranspose", node.inputs, {}, node.name)


def _map_max_pool(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map max pool operation."""
    return builder.make_node("MaxPool", node.inputs, {}, node.name)


def _map_max_pool_3d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map max pool 3d operation."""
    return builder.make_node("MaxPool", node.inputs, {}, node.name)


def _map_avg_pool(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map avg pool operation."""
    return builder.make_node("AveragePool", node.inputs, {}, node.name)


def _map_avg_pool_3d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map avg pool 3d operation."""
    return builder.make_node("AveragePool", node.inputs, {}, node.name)


def _map_global_max_pool(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map global max pool operation."""
    return builder.make_node("GlobalMaxPool", node.inputs, {}, node.name)


def _map_global_avg_pool(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map global avg pool operation."""
    return builder.make_node("GlobalAveragePool", node.inputs, {}, node.name)


def _map_fractional_max_pool(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map fractional max pool operation."""
    return builder.make_node("Custom_FractionalMaxPool", node.inputs, {}, node.name)


def _map_fractional_avg_pool(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map fractional avg pool operation."""
    return builder.make_node("Custom_FractionalAvgPool", node.inputs, {}, node.name)


def _map_batch_norm(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map batch norm operation."""
    epsilon = builder.extract_attr(node, "epsilon", 1e-05)
    return builder.make_node("BatchNormalization", node.inputs, {"epsilon": epsilon}, node.name)


def _map_l2_loss(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map l2 loss operation."""
    return builder.make_node("Custom_L2Loss", node.inputs, {}, node.name)


def _map_lrn(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map lrn operation."""
    return builder.make_node("LRN", node.inputs, {}, node.name)


def _map_dropout(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map dropout operation."""
    return builder.make_node("Dropout", node.inputs, {}, node.name)


def _map_topk(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map topk operation."""
    return builder.make_node("TopK", node.inputs, {}, node.name)


def _map_in_top_k(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map in top k operation."""
    return builder.make_node("Custom_InTopK", node.inputs, {}, node.name)


def _map_nth_element(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map nth element operation."""
    return builder.make_node("Custom_NthElement", node.inputs, {}, node.name)


NN_OPS_MAPPING: dict[str, Callable[[TFToONNXGraphBuilder, TFNode], list[str]]] = {
    "MatMul": _map_simple_binary("MatMul"),
    "BatchMatMul": _map_simple_binary("MatMul"),
    "BatchMatMulV2": _map_simple_binary("MatMul"),
    "BiasAdd": _map_simple_binary("Add"),
    "BiasAddV1": _map_simple_binary("Add"),
    "Relu": _map_simple_unary("Relu"),
    "Relu6": _map_relu6,
    "LeakyRelu": _map_leaky_relu,
    "Elu": _map_elu,
    "Selu": _map_selu,
    "Softplus": _map_softplus,
    "Softsign": _map_softsign,
    "Sigmoid": _map_simple_unary("Sigmoid"),
    "Softmax": _map_softmax,
    "LogSoftmax": _map_log_softmax,
    "Conv2D": _map_conv2d,
    "Conv3D": _map_conv3d,
    "DepthwiseConv2dNative": _map_depthwise_conv2d_native,
    "Conv2DBackpropInput": _map_conv2d_backprop_input,
    "Conv3DBackpropInputV2": _map_conv3d_backprop_input_v2,
    "MaxPool": _map_max_pool,
    "MaxPoolV2": _map_max_pool,
    "MaxPool3D": _map_max_pool_3d,
    "AvgPool": _map_avg_pool,
    "AvgPool3D": _map_avg_pool_3d,
    "GlobalMaxPool": _map_global_max_pool,
    "GlobalAvgPool": _map_global_avg_pool,
    "FractionalMaxPool": _map_fractional_max_pool,
    "FractionalAvgPool": _map_fractional_avg_pool,
    "BatchNormWithGlobalNormalization": _map_batch_norm,
    "FusedBatchNorm": _map_batch_norm,
    "FusedBatchNormV2": _map_batch_norm,
    "FusedBatchNormV3": _map_batch_norm,
    "L2Loss": _map_l2_loss,
    "LocalResponseNormalization": _map_lrn,
    "Dropout": _map_dropout,
    "TopK": _map_topk,
    "TopKV2": _map_topk,
    "InTopK": _map_in_top_k,
    "InTopKV2": _map_in_top_k,
    "NthElement": _map_nth_element,
}
