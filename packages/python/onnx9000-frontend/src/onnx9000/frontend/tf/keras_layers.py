"""Module providing keras layers functionality."""

from typing import Callable
from onnx9000.frontend.tf.builder import TFToONNXGraphBuilder
from onnx9000.frontend.tf.parsers import TFNode


def _map_keras_dense(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras dense operation."""
    return builder.make_node("Custom_KerasDense", node.inputs, node.attr, node.name)


def _map_keras_conv1d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras conv1d operation."""
    return builder.make_node("Custom_KerasConv1D", node.inputs, node.attr, node.name)


def _map_keras_conv2d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras conv2d operation."""
    return builder.make_node("Custom_KerasConv2D", node.inputs, node.attr, node.name)


def _map_keras_conv3d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras conv3d operation."""
    return builder.make_node("Custom_KerasConv3D", node.inputs, node.attr, node.name)


def _map_keras_separable_conv1d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras separable conv1d operation."""
    return builder.make_node("Custom_KerasSeparableConv1D", node.inputs, node.attr, node.name)


def _map_keras_separable_conv2d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras separable conv2d operation."""
    return builder.make_node("Custom_KerasSeparableConv2D", node.inputs, node.attr, node.name)


def _map_keras_depthwise_conv2d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras depthwise conv2d operation."""
    return builder.make_node("Custom_KerasDepthwiseConv2D", node.inputs, node.attr, node.name)


def _map_keras_conv2d_transpose(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras conv2d transpose operation."""
    return builder.make_node("Custom_KerasConv2DTranspose", node.inputs, node.attr, node.name)


def _map_keras_conv3d_transpose(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras conv3d transpose operation."""
    return builder.make_node("Custom_KerasConv3DTranspose", node.inputs, node.attr, node.name)


def _map_keras_max_pooling_1d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras max pooling 1d operation."""
    return builder.make_node("Custom_KerasMaxPooling1D", node.inputs, node.attr, node.name)


def _map_keras_max_pooling_2d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras max pooling 2d operation."""
    return builder.make_node("Custom_KerasMaxPooling2D", node.inputs, node.attr, node.name)


def _map_keras_max_pooling_3d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras max pooling 3d operation."""
    return builder.make_node("Custom_KerasMaxPooling3D", node.inputs, node.attr, node.name)


def _map_keras_avg_pooling_1d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras avg pooling 1d operation."""
    return builder.make_node("Custom_KerasAvgPooling1D", node.inputs, node.attr, node.name)


def _map_keras_avg_pooling_2d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras avg pooling 2d operation."""
    return builder.make_node("Custom_KerasAvgPooling2D", node.inputs, node.attr, node.name)


def _map_keras_avg_pooling_3d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras avg pooling 3d operation."""
    return builder.make_node("Custom_KerasAvgPooling3D", node.inputs, node.attr, node.name)


def _map_keras_global_max_pooling_1d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras global max pooling 1d operation."""
    return builder.make_node("Custom_KerasGlobalMaxPooling1D", node.inputs, node.attr, node.name)


def _map_keras_global_max_pooling_2d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras global max pooling 2d operation."""
    return builder.make_node("Custom_KerasGlobalMaxPooling2D", node.inputs, node.attr, node.name)


def _map_keras_global_max_pooling_3d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras global max pooling 3d operation."""
    return builder.make_node("Custom_KerasGlobalMaxPooling3D", node.inputs, node.attr, node.name)


def _map_keras_global_avg_pooling_1d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras global avg pooling 1d operation."""
    return builder.make_node("Custom_KerasGlobalAvgPooling1D", node.inputs, node.attr, node.name)


def _map_keras_global_avg_pooling_2d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras global avg pooling 2d operation."""
    return builder.make_node("Custom_KerasGlobalAvgPooling2D", node.inputs, node.attr, node.name)


def _map_keras_global_avg_pooling_3d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras global avg pooling 3d operation."""
    return builder.make_node("Custom_KerasGlobalAvgPooling3D", node.inputs, node.attr, node.name)


def _map_keras_rnn(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras rnn operation."""
    return builder.make_node("Custom_KerasRNN", node.inputs, node.attr, node.name)


def _map_keras_simple_rnn(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras simple rnn operation."""
    return builder.make_node("RNN", node.inputs, node.attr, node.name)


def _map_keras_lstm(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras lstm operation."""
    return builder.make_node("LSTM", node.inputs, node.attr, node.name)


def _map_keras_gru(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras gru operation."""
    return builder.make_node("GRU", node.inputs, node.attr, node.name)


def _map_keras_bidirectional(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras bidirectional operation."""
    return builder.make_node("Custom_KerasBidirectional", node.inputs, node.attr, node.name)


def _map_keras_embedding(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras embedding operation."""
    return builder.make_node("Gather", node.inputs, node.attr, node.name)


def _map_keras_batch_normalization(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras batch normalization operation."""
    return builder.make_node("BatchNormalization", node.inputs, node.attr, node.name)


def _map_keras_layer_normalization(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras layer normalization operation."""
    return builder.make_node("LayerNormalization", node.inputs, node.attr, node.name)


def _map_keras_dropout(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras dropout operation."""
    return builder.make_node("Dropout", node.inputs, node.attr, node.name)


def _map_keras_spatial_dropout(dim: str) -> Callable:
    """Executes the  map keras spatial dropout operation."""

    def _impl(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
        """Executes the  impl operation."""
        return builder.make_node(
            f"Custom_KerasSpatialDropout{dim}", node.inputs, node.attr, node.name
        )

    return _impl


def _map_keras_activation(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras activation operation."""
    return builder.make_node("Custom_KerasActivation", node.inputs, node.attr, node.name)


def _map_keras_flatten(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras flatten operation."""
    return builder.make_node("Flatten", node.inputs, node.attr, node.name)


def _map_keras_reshape(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras reshape operation."""
    return builder.make_node("Reshape", node.inputs, node.attr, node.name)


def _map_keras_permute(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras permute operation."""
    return builder.make_node("Transpose", node.inputs, node.attr, node.name)


def _map_keras_repeat_vector(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras repeat vector operation."""
    return builder.make_node("Custom_KerasRepeatVector", node.inputs, node.attr, node.name)


def _map_keras_cropping(dim: str) -> Callable:
    """Executes the  map keras cropping operation."""

    def _impl(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
        """Executes the  impl operation."""
        return builder.make_node(f"Custom_KerasCropping{dim}", node.inputs, node.attr, node.name)

    return _impl


def _map_keras_upsampling(dim: str) -> Callable:
    """Executes the  map keras upsampling operation."""

    def _impl(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
        """Executes the  impl operation."""
        return builder.make_node(f"Custom_KerasUpSampling{dim}", node.inputs, node.attr, node.name)

    return _impl


def _map_keras_zero_padding(dim: str) -> Callable:
    """Executes the  map keras zero padding operation."""

    def _impl(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
        """Executes the  impl operation."""
        return builder.make_node(f"Custom_KerasZeroPadding{dim}", node.inputs, node.attr, node.name)

    return _impl


def _map_keras_concatenate(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras concatenate operation."""
    return builder.make_node("Concat", node.inputs, node.attr, node.name)


def _map_keras_average(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras average operation."""
    return builder.make_node("Mean", node.inputs, node.attr, node.name)


def _map_keras_maximum(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras maximum operation."""
    return builder.make_node("Max", node.inputs, node.attr, node.name)


def _map_keras_minimum(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras minimum operation."""
    return builder.make_node("Min", node.inputs, node.attr, node.name)


def _map_keras_add(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras add operation."""
    return builder.make_node("Sum", node.inputs, node.attr, node.name)


def _map_keras_subtract(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras subtract operation."""
    return builder.make_node("Sub", node.inputs, node.attr, node.name)


def _map_keras_multiply(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras multiply operation."""
    return builder.make_node("Mul", node.inputs, node.attr, node.name)


def _map_keras_dot(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Executes the  map keras dot operation."""
    return builder.make_node("Custom_KerasDot", node.inputs, node.attr, node.name)


KERAS_LAYERS_MAPPING: dict[str, Callable[[TFToONNXGraphBuilder, TFNode], list[str]]] = {
    "Dense": _map_keras_dense,
    "Conv1D": _map_keras_conv1d,
    "Conv2D": _map_keras_conv2d,
    "Conv3D": _map_keras_conv3d,
    "SeparableConv1D": _map_keras_separable_conv1d,
    "SeparableConv2D": _map_keras_separable_conv2d,
    "DepthwiseConv2D": _map_keras_depthwise_conv2d,
    "Conv2DTranspose": _map_keras_conv2d_transpose,
    "Conv3DTranspose": _map_keras_conv3d_transpose,
    "MaxPooling1D": _map_keras_max_pooling_1d,
    "MaxPooling2D": _map_keras_max_pooling_2d,
    "MaxPooling3D": _map_keras_max_pooling_3d,
    "AveragePooling1D": _map_keras_avg_pooling_1d,
    "AveragePooling2D": _map_keras_avg_pooling_2d,
    "AveragePooling3D": _map_keras_avg_pooling_3d,
    "GlobalMaxPooling1D": _map_keras_global_max_pooling_1d,
    "GlobalMaxPooling2D": _map_keras_global_max_pooling_2d,
    "GlobalMaxPooling3D": _map_keras_global_max_pooling_3d,
    "GlobalAveragePooling1D": _map_keras_global_avg_pooling_1d,
    "GlobalAveragePooling2D": _map_keras_global_avg_pooling_2d,
    "GlobalAveragePooling3D": _map_keras_global_avg_pooling_3d,
    "RNN": _map_keras_rnn,
    "SimpleRNN": _map_keras_simple_rnn,
    "LSTM": _map_keras_lstm,
    "GRU": _map_keras_gru,
    "Bidirectional": _map_keras_bidirectional,
    "Embedding": _map_keras_embedding,
    "BatchNormalization": _map_keras_batch_normalization,
    "LayerNormalization": _map_keras_layer_normalization,
    "Dropout": _map_keras_dropout,
    "SpatialDropout1D": _map_keras_spatial_dropout("1D"),
    "SpatialDropout2D": _map_keras_spatial_dropout("2D"),
    "SpatialDropout3D": _map_keras_spatial_dropout("3D"),
    "Activation": _map_keras_activation,
    "Flatten": _map_keras_flatten,
    "Reshape": _map_keras_reshape,
    "Permute": _map_keras_permute,
    "RepeatVector": _map_keras_repeat_vector,
    "Cropping1D": _map_keras_cropping("1D"),
    "Cropping2D": _map_keras_cropping("2D"),
    "Cropping3D": _map_keras_cropping("3D"),
    "UpSampling1D": _map_keras_upsampling("1D"),
    "UpSampling2D": _map_keras_upsampling("2D"),
    "UpSampling3D": _map_keras_upsampling("3D"),
    "ZeroPadding1D": _map_keras_zero_padding("1D"),
    "ZeroPadding2D": _map_keras_zero_padding("2D"),
    "ZeroPadding3D": _map_keras_zero_padding("3D"),
    "Concatenate": _map_keras_concatenate,
    "Average": _map_keras_average,
    "Maximum": _map_keras_maximum,
    "Minimum": _map_keras_minimum,
    "Add": _map_keras_add,
    "Subtract": _map_keras_subtract,
    "Multiply": _map_keras_multiply,
    "Dot": _map_keras_dot,
}
