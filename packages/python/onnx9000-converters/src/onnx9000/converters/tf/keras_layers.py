"""Module providing keras layers functionality."""

from typing import Callable

from onnx9000.converters.tf.builder import TFToONNXGraphBuilder
from onnx9000.converters.tf.parsers import TFNode


def _map_keras_dense(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras dense operation."""
    if len(node.inputs) == 3:
        mm = builder.make_node("MatMul", [node.inputs[0], node.inputs[1]], {}, f"{node.name}_mm")[0]
        return builder.make_node("Add", [mm, node.inputs[2]], node.attr, node.name)
    return builder.make_node("MatMul", node.inputs[:2], node.attr, node.name)


def _map_keras_conv1d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras conv1d operation."""
    return builder.make_node("Conv", node.inputs, node.attr, node.name)


def _map_keras_conv2d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras conv2d operation."""
    return builder.make_node("Conv", node.inputs, node.attr, node.name)


def _map_keras_conv3d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras conv3d operation."""
    return builder.make_node("Conv", node.inputs, node.attr, node.name)


def _map_keras_separable_conv1d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras separable conv1d operation."""
    return builder.make_node("Conv", node.inputs, node.attr, node.name)


def _map_keras_separable_conv2d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras separable conv2d operation."""
    return builder.make_node("Conv", node.inputs, node.attr, node.name)


def _map_keras_depthwise_conv2d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras depthwise conv2d operation."""
    return builder.make_node("Conv", node.inputs, node.attr, node.name)


def _map_keras_conv2d_transpose(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras conv2d transpose operation."""
    return builder.make_node("ConvTranspose", node.inputs, node.attr, node.name)


def _map_keras_conv3d_transpose(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras conv3d transpose operation."""
    return builder.make_node("ConvTranspose", node.inputs, node.attr, node.name)


def _map_keras_max_pooling_1d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras max pooling 1d operation."""
    return builder.make_node("MaxPool", node.inputs, node.attr, node.name)


def _map_keras_max_pooling_2d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras max pooling 2d operation."""
    return builder.make_node("MaxPool", node.inputs, node.attr, node.name)


def _map_keras_max_pooling_3d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras max pooling 3d operation."""
    return builder.make_node("MaxPool", node.inputs, node.attr, node.name)


def _map_keras_avg_pooling_1d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras avg pooling 1d operation."""
    return builder.make_node("AveragePool", node.inputs, node.attr, node.name)


def _map_keras_avg_pooling_2d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras avg pooling 2d operation."""
    return builder.make_node("AveragePool", node.inputs, node.attr, node.name)


def _map_keras_avg_pooling_3d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras avg pooling 3d operation."""
    return builder.make_node("AveragePool", node.inputs, node.attr, node.name)


def _map_keras_global_max_pooling_1d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras global max pooling 1d operation."""
    return builder.make_node("GlobalMaxPool", node.inputs, node.attr, node.name)


def _map_keras_global_max_pooling_2d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras global max pooling 2d operation."""
    return builder.make_node("GlobalMaxPool", node.inputs, node.attr, node.name)


def _map_keras_global_max_pooling_3d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras global max pooling 3d operation."""
    return builder.make_node("GlobalMaxPool", node.inputs, node.attr, node.name)


def _map_keras_global_avg_pooling_1d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras global avg pooling 1d operation."""
    return builder.make_node("GlobalAveragePool", node.inputs, node.attr, node.name)


def _map_keras_global_avg_pooling_2d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras global avg pooling 2d operation."""
    return builder.make_node("GlobalAveragePool", node.inputs, node.attr, node.name)


def _map_keras_global_avg_pooling_3d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras global avg pooling 3d operation."""
    return builder.make_node("GlobalAveragePool", node.inputs, node.attr, node.name)


def _map_keras_rnn(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras rnn operation."""
    return builder.make_node("RNN", node.inputs, node.attr, node.name)


def _map_keras_simple_rnn(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras simple rnn operation."""
    return builder.make_node("RNN", node.inputs, node.attr, node.name)


def _map_keras_lstm(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras lstm operation."""
    return builder.make_node("LSTM", node.inputs, node.attr, node.name)


def _map_keras_gru(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras gru operation."""
    return builder.make_node("GRU", node.inputs, node.attr, node.name)


def _map_keras_bidirectional(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras bidirectional operation."""
    return builder.make_node("RNN", node.inputs, node.attr, node.name)


def _map_keras_embedding(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras embedding operation."""
    return builder.make_node("Gather", node.inputs, node.attr, node.name)


def _map_keras_batch_normalization(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras batch normalization operation."""
    return builder.make_node("BatchNormalization", node.inputs, node.attr, node.name)


def _map_keras_layer_normalization(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layer normalization operation."""
    return builder.make_node("LayerNormalization", node.inputs, node.attr, node.name)


def _map_keras_dropout(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras dropout operation."""
    return builder.make_node("Dropout", node.inputs, node.attr, node.name)


def _map_keras_spatial_dropout(dim: str) -> Callable:
    """Execute the  map keras spatial dropout operation."""

    def _impl(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
        """Execute the  impl operation."""
        return builder.make_node("Dropout", node.inputs, node.attr, node.name)

    return _impl


def _map_keras_activation(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras activation operation."""
    act = (
        node.attr.get("activation", b"linear").decode("utf-8")
        if isinstance(node.attr.get("activation"), bytes)
        else node.attr.get("activation", "linear")
    )
    op_map = {
        "relu": "Relu",
        "sigmoid": "Sigmoid",
        "tanh": "Tanh",
        "softmax": "Softmax",
        "linear": "Identity",
    }
    return builder.make_node(op_map.get(act, "Relu"), node.inputs, node.attr, node.name)


def _map_keras_flatten(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras flatten operation."""
    return builder.make_node("Flatten", node.inputs, node.attr, node.name)


def _map_keras_reshape(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras reshape operation."""
    return builder.make_node("Reshape", node.inputs, node.attr, node.name)


def _map_keras_permute(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras permute operation."""
    return builder.make_node("Transpose", node.inputs, node.attr, node.name)


def _map_keras_repeat_vector(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras repeat vector operation."""
    return builder.make_node("Tile", node.inputs, node.attr, node.name)


def _map_keras_cropping(dim: str) -> Callable:
    """Execute the  map keras cropping operation."""

    def _impl(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
        """Execute the  impl operation."""
        return builder.make_node("Slice", node.inputs, node.attr, node.name)

    return _impl


def _map_keras_upsampling(dim: str) -> Callable:
    """Execute the  map keras upsampling operation."""

    def _impl(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
        """Execute the  impl operation."""
        return builder.make_node("Resize", node.inputs, node.attr, node.name)

    return _impl


def _map_keras_zero_padding(dim: str) -> Callable:
    """Execute the  map keras zero padding operation."""

    def _impl(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
        """Execute the  impl operation."""
        return builder.make_node("Pad", node.inputs, node.attr, node.name)

    return _impl


def _map_keras_concatenate(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras concatenate operation."""
    return builder.make_node("Concat", node.inputs, node.attr, node.name)


def _map_keras_average(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras average operation."""
    return builder.make_node("Mean", node.inputs, node.attr, node.name)


def _map_keras_maximum(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras maximum operation."""
    return builder.make_node("Max", node.inputs, node.attr, node.name)


def _map_keras_minimum(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras minimum operation."""
    return builder.make_node("Min", node.inputs, node.attr, node.name)


def _map_keras_add(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras add operation."""
    return builder.make_node("Sum", node.inputs, node.attr, node.name)


def _map_keras_subtract(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras subtract operation."""
    return builder.make_node("Sub", node.inputs, node.attr, node.name)


def _map_keras_multiply(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras multiply operation."""
    return builder.make_node("Mul", node.inputs, node.attr, node.name)


def _map_keras_dot(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras dot operation."""
    return builder.make_node("MatMul", node.inputs, node.attr, node.name)


def _map_keras_variable(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras Variable operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_device(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras device operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_name_scope(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras name_scope operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_keras_tensor(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras KerasTensor operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_remat_scope(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras RematScope operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_remat(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras remat operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_stateless_scope(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras StatelessScope operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_symbolic_scope(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras SymbolicScope operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_d_type_policy(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras DTypePolicy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_float_d_type_policy(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras FloatDTypePolicy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_initializer(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras Initializer operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_input(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras Input operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_input_spec(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras InputSpec operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layer_ext(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras Layer operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_loss(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras Loss operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metric(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras Metric operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_model(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras Model operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_sequential(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras Sequential operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_function(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras Function operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_operation(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras Operation operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_optimizer(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras Optimizer operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_quantizer(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras Quantizer operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_regularizer(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras Regularizer operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_version(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras version operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_saving__keras_file_editor(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras saving.KerasFileEditor operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_saving__custom_object_scope(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras saving.CustomObjectScope operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_saving_custom_object_scope(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras saving.custom_object_scope operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_saving_get_custom_objects(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras saving.get_custom_objects operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_saving_get_registered_name(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras saving.get_registered_name operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_saving_get_registered_object(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras saving.get_registered_object operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_saving_register_keras_serializable(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras saving.register_keras_serializable operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_saving_load_model(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras saving.load_model operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_saving_load_weights(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras saving.load_weights operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_saving_save_model(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras saving.save_model operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_saving_save_weights(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras saving.save_weights operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_saving_deserialize_keras_object(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras saving.deserialize_keras_object operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_saving_serialize_keras_object(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras saving.serialize_keras_object operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_export__export_archive(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras export.ExportArchive operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_preprocessing_image_dataset_from_directory(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras preprocessing.image_dataset_from_directory operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_preprocessing_text_dataset_from_directory(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras preprocessing.text_dataset_from_directory operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_preprocessing_timeseries_dataset_from_array(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras preprocessing.timeseries_dataset_from_array operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_preprocessing_image_array_to_img(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras preprocessing.image.array_to_img operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_preprocessing_image_img_to_array(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras preprocessing.image.img_to_array operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_preprocessing_image_load_img(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras preprocessing.image.load_img operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_preprocessing_image_save_img(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras preprocessing.image.save_img operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_preprocessing_image_smart_resize(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras preprocessing.image.smart_resize operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_preprocessing_sequence_pad_sequences(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras preprocessing.sequence.pad_sequences operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_distillation__distillation_loss(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras distillation.DistillationLoss operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_distillation__feature_distillation(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras distillation.FeatureDistillation operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_distillation__logits_distillation(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras distillation.LogitsDistillation operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_distillation__distiller(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras distillation.Distiller operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_models_clone_model(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras models.clone_model operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_models__model(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras models.Model operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_models_model_from_json(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras models.model_from_json operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_models__sequential(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras models.Sequential operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_models_load_model(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras models.load_model operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_models_save_model(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras models.save_model operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications__conv_ne_xt_base(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.ConvNeXtBase operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications__conv_ne_xt_large(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.ConvNeXtLarge operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications__conv_ne_xt_small(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.ConvNeXtSmall operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications__conv_ne_xt_tiny(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.ConvNeXtTiny operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications__conv_ne_xt_x_large(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.ConvNeXtXLarge operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications__dense_net121(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras applications.DenseNet121 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications__dense_net169(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras applications.DenseNet169 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications__dense_net201(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras applications.DenseNet201 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications__efficient_net_b0(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.EfficientNetB0 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications__efficient_net_b1(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.EfficientNetB1 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications__efficient_net_b2(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.EfficientNetB2 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications__efficient_net_b3(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.EfficientNetB3 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications__efficient_net_b4(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.EfficientNetB4 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications__efficient_net_b5(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.EfficientNetB5 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications__efficient_net_b6(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.EfficientNetB6 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications__efficient_net_b7(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.EfficientNetB7 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications__efficient_net_v2_b0(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.EfficientNetV2B0 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications__efficient_net_v2_b1(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.EfficientNetV2B1 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications__efficient_net_v2_b2(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.EfficientNetV2B2 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications__efficient_net_v2_b3(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.EfficientNetV2B3 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications__efficient_net_v2_l(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.EfficientNetV2L operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications__efficient_net_v2_m(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.EfficientNetV2M operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications__efficient_net_v2_s(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.EfficientNetV2S operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications__inception_res_net_v2(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.InceptionResNetV2 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications__inception_v3(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras applications.InceptionV3 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications__mobile_net(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras applications.MobileNet operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications__mobile_net_v2(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.MobileNetV2 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications__mobile_net_v3_large(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.MobileNetV3Large operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications__mobile_net_v3_small(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.MobileNetV3Small operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_nas_net_large(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras applications.NASNetLarge operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_nas_net_mobile(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.NASNetMobile operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications__res_net50(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras applications.ResNet50 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications__res_net101(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras applications.ResNet101 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications__res_net152(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras applications.ResNet152 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications__res_net50_v2(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras applications.ResNet50V2 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications__res_net101_v2(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.ResNet101V2 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications__res_net152_v2(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.ResNet152V2 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_vgg16(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras applications.VGG16 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_vgg19(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras applications.VGG19 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications__xception(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras applications.Xception operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_mobilenet_v3_decode_predictions(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.mobilenet_v3.decode_predictions operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_mobilenet_v3_preprocess_input(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.mobilenet_v3.preprocess_input operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_inception_resnet_v2__inception_res_net_v2(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.inception_resnet_v2.InceptionResNetV2 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_inception_resnet_v2_decode_predictions(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.inception_resnet_v2.decode_predictions operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_inception_resnet_v2_preprocess_input(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.inception_resnet_v2.preprocess_input operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_resnet_v2__res_net50_v2(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.resnet_v2.ResNet50V2 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_resnet_v2__res_net101_v2(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.resnet_v2.ResNet101V2 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_resnet_v2__res_net152_v2(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.resnet_v2.ResNet152V2 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_resnet_v2_decode_predictions(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.resnet_v2.decode_predictions operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_resnet_v2_preprocess_input(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.resnet_v2.preprocess_input operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_convnext__conv_ne_xt_base(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.convnext.ConvNeXtBase operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_convnext__conv_ne_xt_large(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.convnext.ConvNeXtLarge operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_convnext__conv_ne_xt_small(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.convnext.ConvNeXtSmall operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_convnext__conv_ne_xt_tiny(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.convnext.ConvNeXtTiny operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_convnext__conv_ne_xt_x_large(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.convnext.ConvNeXtXLarge operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_convnext_decode_predictions(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.convnext.decode_predictions operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_convnext_preprocess_input(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.convnext.preprocess_input operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_inception_v3__inception_v3(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.inception_v3.InceptionV3 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_inception_v3_decode_predictions(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.inception_v3.decode_predictions operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_inception_v3_preprocess_input(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.inception_v3.preprocess_input operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_xception__xception(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.xception.Xception operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_xception_decode_predictions(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.xception.decode_predictions operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_xception_preprocess_input(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.xception.preprocess_input operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_resnet50__res_net50(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.resnet50.ResNet50 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_resnet50_decode_predictions(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.resnet50.decode_predictions operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_resnet50_preprocess_input(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.resnet50.preprocess_input operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_vgg19_vgg19(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras applications.vgg19.VGG19 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_vgg19_decode_predictions(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.vgg19.decode_predictions operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_vgg19_preprocess_input(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.vgg19.preprocess_input operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_resnet__res_net50(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.resnet.ResNet50 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_resnet__res_net101(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.resnet.ResNet101 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_resnet__res_net152(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.resnet.ResNet152 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_resnet_decode_predictions(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.resnet.decode_predictions operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_resnet_preprocess_input(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.resnet.preprocess_input operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_vgg16_vgg16(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras applications.vgg16.VGG16 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_vgg16_decode_predictions(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.vgg16.decode_predictions operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_vgg16_preprocess_input(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.vgg16.preprocess_input operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_densenet__dense_net121(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.densenet.DenseNet121 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_densenet__dense_net169(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.densenet.DenseNet169 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_densenet__dense_net201(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.densenet.DenseNet201 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_densenet_decode_predictions(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.densenet.decode_predictions operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_densenet_preprocess_input(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.densenet.preprocess_input operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_nasnet_nas_net_large(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.nasnet.NASNetLarge operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_nasnet_nas_net_mobile(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.nasnet.NASNetMobile operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_nasnet_decode_predictions(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.nasnet.decode_predictions operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_nasnet_preprocess_input(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.nasnet.preprocess_input operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_mobilenet__mobile_net(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.mobilenet.MobileNet operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_mobilenet_decode_predictions(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.mobilenet.decode_predictions operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_mobilenet_preprocess_input(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.mobilenet.preprocess_input operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_imagenet_utils_decode_predictions(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.imagenet_utils.decode_predictions operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_imagenet_utils_preprocess_input(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.imagenet_utils.preprocess_input operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_efficientnet__efficient_net_b0(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.efficientnet.EfficientNetB0 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_efficientnet__efficient_net_b1(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.efficientnet.EfficientNetB1 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_efficientnet__efficient_net_b2(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.efficientnet.EfficientNetB2 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_efficientnet__efficient_net_b3(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.efficientnet.EfficientNetB3 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_efficientnet__efficient_net_b4(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.efficientnet.EfficientNetB4 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_efficientnet__efficient_net_b5(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.efficientnet.EfficientNetB5 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_efficientnet__efficient_net_b6(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.efficientnet.EfficientNetB6 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_efficientnet__efficient_net_b7(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.efficientnet.EfficientNetB7 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_efficientnet_decode_predictions(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.efficientnet.decode_predictions operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_efficientnet_preprocess_input(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.efficientnet.preprocess_input operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_efficientnet_v2__efficient_net_v2_b0(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.efficientnet_v2.EfficientNetV2B0 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_efficientnet_v2__efficient_net_v2_b1(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.efficientnet_v2.EfficientNetV2B1 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_efficientnet_v2__efficient_net_v2_b2(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.efficientnet_v2.EfficientNetV2B2 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_efficientnet_v2__efficient_net_v2_b3(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.efficientnet_v2.EfficientNetV2B3 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_efficientnet_v2__efficient_net_v2_l(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.efficientnet_v2.EfficientNetV2L operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_efficientnet_v2__efficient_net_v2_m(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.efficientnet_v2.EfficientNetV2M operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_efficientnet_v2__efficient_net_v2_s(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.efficientnet_v2.EfficientNetV2S operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_efficientnet_v2_decode_predictions(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.efficientnet_v2.decode_predictions operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_efficientnet_v2_preprocess_input(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.efficientnet_v2.preprocess_input operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_mobilenet_v2__mobile_net_v2(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.mobilenet_v2.MobileNetV2 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_mobilenet_v2_decode_predictions(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.mobilenet_v2.decode_predictions operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_applications_mobilenet_v2_preprocess_input(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras applications.mobilenet_v2.preprocess_input operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_initializers_deserialize(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras initializers.deserialize operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_initializers_get(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras initializers.get operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_initializers_serialize(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras initializers.serialize operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_initializers_stft(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras initializers.STFT operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_initializers_stft_initializer(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras initializers.STFTInitializer operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_initializers_stft_ext(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras initializers.stft operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_initializers__constant(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras initializers.Constant operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_initializers_constant(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras initializers.constant operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_initializers__identity(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras initializers.Identity operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_initializers__identity_initializer(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras initializers.IdentityInitializer operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_initializers_identity(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras initializers.identity operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_initializers__ones(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras initializers.Ones operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_initializers_ones(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras initializers.ones operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_initializers__zeros(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras initializers.Zeros operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_initializers_zeros(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras initializers.zeros operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_initializers__initializer(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras initializers.Initializer operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_initializers__glorot_normal(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras initializers.GlorotNormal operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_initializers_glorot_normal(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras initializers.glorot_normal operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_initializers__glorot_uniform(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras initializers.GlorotUniform operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_initializers_glorot_uniform(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras initializers.glorot_uniform operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_initializers__he_normal(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras initializers.HeNormal operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_initializers_he_normal(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras initializers.he_normal operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_initializers__he_uniform(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras initializers.HeUniform operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_initializers_he_uniform(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras initializers.he_uniform operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_initializers__lecun_normal(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras initializers.LecunNormal operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_initializers_lecun_normal(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras initializers.lecun_normal operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_initializers__lecun_uniform(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras initializers.LecunUniform operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_initializers_lecun_uniform(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras initializers.lecun_uniform operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_initializers__orthogonal(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras initializers.Orthogonal operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_initializers__orthogonal_initializer(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras initializers.OrthogonalInitializer operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_initializers_orthogonal(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras initializers.orthogonal operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_initializers__random_normal(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras initializers.RandomNormal operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_initializers_random_normal(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras initializers.random_normal operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_initializers__random_uniform(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras initializers.RandomUniform operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_initializers_random_uniform(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras initializers.random_uniform operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_initializers__truncated_normal(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras initializers.TruncatedNormal operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_initializers_truncated_normal(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras initializers.truncated_normal operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_initializers__variance_scaling(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras initializers.VarianceScaling operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_initializers_variance_scaling(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras initializers.variance_scaling operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_quantizers_deserialize(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras quantizers.deserialize operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_quantizers_get(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras quantizers.get operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_quantizers_serialize(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras quantizers.serialize operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_quantizers_gptq_config(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras quantizers.GPTQConfig operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_quantizers__float8_quantization_config(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras quantizers.Float8QuantizationConfig operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_quantizers__int4_quantization_config(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras quantizers.Int4QuantizationConfig operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_quantizers__int8_quantization_config(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras quantizers.Int8QuantizationConfig operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_quantizers__quantization_config(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras quantizers.QuantizationConfig operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_quantizers__abs_max_quantizer(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras quantizers.AbsMaxQuantizer operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_quantizers__quantizer(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras quantizers.Quantizer operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_quantizers_abs_max_quantize(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras quantizers.abs_max_quantize operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_quantizers_compute_float8_amax_history(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras quantizers.compute_float8_amax_history operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_quantizers_compute_float8_scale(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras quantizers.compute_float8_scale operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_quantizers_fake_quant_with_min_max_vars(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras quantizers.fake_quant_with_min_max_vars operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_quantizers_pack_int4(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras quantizers.pack_int4 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_quantizers_quantize_and_dequantize(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras quantizers.quantize_and_dequantize operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_quantizers_unpack_int4(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras quantizers.unpack_int4 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_deserialize(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras activations.deserialize operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_get(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras activations.get operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_serialize(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras activations.serialize operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_celu(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras activations.celu operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_elu(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras activations.elu operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_exponential(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras activations.exponential operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_gelu(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras activations.gelu operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_glu(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras activations.glu operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_hard_shrink(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras activations.hard_shrink operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_hard_sigmoid(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras activations.hard_sigmoid operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_hard_silu(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras activations.hard_silu operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_hard_swish(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras activations.hard_swish operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_hard_tanh(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras activations.hard_tanh operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_leaky_relu(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras activations.leaky_relu operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_linear(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras activations.linear operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_log_sigmoid(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras activations.log_sigmoid operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_log_softmax(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras activations.log_softmax operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_mish(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras activations.mish operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_relu(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras activations.relu operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_relu6(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras activations.relu6 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_selu(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras activations.selu operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_sigmoid(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras activations.sigmoid operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_silu(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras activations.silu operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_swish(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras activations.swish operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_soft_shrink(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras activations.soft_shrink operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_softmax(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras activations.softmax operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_softplus(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras activations.softplus operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_softsign(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras activations.softsign operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_sparse_plus(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras activations.sparse_plus operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_sparse_sigmoid(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras activations.sparse_sigmoid operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_sparsemax(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras activations.sparsemax operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_squareplus(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras activations.squareplus operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_tanh(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras activations.tanh operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_tanh_shrink(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras activations.tanh_shrink operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_threshold(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras activations.threshold operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics_binary_crossentropy(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras metrics.binary_crossentropy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics_binary_focal_crossentropy(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras metrics.binary_focal_crossentropy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics_categorical_crossentropy(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras metrics.categorical_crossentropy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics_categorical_focal_crossentropy(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras metrics.categorical_focal_crossentropy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics_categorical_hinge(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras metrics.categorical_hinge operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics_hinge(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras metrics.hinge operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics_huber(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras metrics.huber operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics_kl_divergence(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras metrics.kl_divergence operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics_log_cosh(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras metrics.log_cosh operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics_mean_absolute_error(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras metrics.mean_absolute_error operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics_mean_absolute_percentage_error(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras metrics.mean_absolute_percentage_error operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics_mean_squared_error(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras metrics.mean_squared_error operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics_mean_squared_logarithmic_error(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras metrics.mean_squared_logarithmic_error operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics_poisson(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras metrics.poisson operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics_sparse_categorical_crossentropy(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras metrics.sparse_categorical_crossentropy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics_squared_hinge(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras metrics.squared_hinge operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics_deserialize(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras metrics.deserialize operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics_get(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras metrics.get operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics_serialize(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras metrics.serialize operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics__accuracy(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras metrics.Accuracy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics__binary_accuracy(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras metrics.BinaryAccuracy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics__categorical_accuracy(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras metrics.CategoricalAccuracy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics__sparse_categorical_accuracy(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras metrics.SparseCategoricalAccuracy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics__sparse_top_k_categorical_accuracy(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras metrics.SparseTopKCategoricalAccuracy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics__top_k_categorical_accuracy(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras metrics.TopKCategoricalAccuracy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics_binary_accuracy(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras metrics.binary_accuracy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics_categorical_accuracy(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras metrics.categorical_accuracy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics_sparse_categorical_accuracy(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras metrics.sparse_categorical_accuracy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics_sparse_top_k_categorical_accuracy(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras metrics.sparse_top_k_categorical_accuracy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics_top_k_categorical_accuracy(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras metrics.top_k_categorical_accuracy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics_auc(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras metrics.AUC operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics__false_negatives(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras metrics.FalseNegatives operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics__false_positives(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras metrics.FalsePositives operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics__precision(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras metrics.Precision operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics__precision_at_recall(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras metrics.PrecisionAtRecall operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics__recall(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras metrics.Recall operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics__recall_at_precision(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras metrics.RecallAtPrecision operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics__sensitivity_at_specificity(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras metrics.SensitivityAtSpecificity operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics__specificity_at_sensitivity(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras metrics.SpecificityAtSensitivity operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics__true_negatives(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras metrics.TrueNegatives operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics__true_positives(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras metrics.TruePositives operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics__concordance_correlation(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras metrics.ConcordanceCorrelation operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics__pearson_correlation(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras metrics.PearsonCorrelation operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics_concordance_correlation(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras metrics.concordance_correlation operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics_pearson_correlation(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras metrics.pearson_correlation operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics_f1_score(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras metrics.F1Score operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics_f_beta_score(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras metrics.FBetaScore operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics__categorical_hinge(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras metrics.CategoricalHinge operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics__hinge(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras metrics.Hinge operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics__squared_hinge(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras metrics.SquaredHinge operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics__binary_io_u(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras metrics.BinaryIoU operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics__io_u(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras metrics.IoU operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics__mean_io_u(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras metrics.MeanIoU operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics__one_hot_io_u(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras metrics.OneHotIoU operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics__one_hot_mean_io_u(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras metrics.OneHotMeanIoU operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics__metric(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras metrics.Metric operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics__binary_crossentropy(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras metrics.BinaryCrossentropy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics__categorical_crossentropy(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras metrics.CategoricalCrossentropy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics_kl_divergence_ext(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras metrics.KLDivergence operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics__poisson(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras metrics.Poisson operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics__sparse_categorical_crossentropy(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras metrics.SparseCategoricalCrossentropy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics__mean(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras metrics.Mean operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics__mean_metric_wrapper(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras metrics.MeanMetricWrapper operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics__sum(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras metrics.Sum operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics__cosine_similarity(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras metrics.CosineSimilarity operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics__log_cosh_error(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras metrics.LogCoshError operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics__mean_absolute_error(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras metrics.MeanAbsoluteError operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics__mean_absolute_percentage_error(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras metrics.MeanAbsolutePercentageError operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics__mean_squared_error(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras metrics.MeanSquaredError operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics__mean_squared_logarithmic_error(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras metrics.MeanSquaredLogarithmicError operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics_r2_score(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras metrics.R2Score operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_metrics__root_mean_squared_error(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras metrics.RootMeanSquaredError operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses_deserialize(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras losses.deserialize operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses_get(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras losses.get operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses_serialize(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras losses.serialize operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses__loss(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras losses.Loss operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses_ctc(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras losses.CTC operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses__binary_crossentropy(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras losses.BinaryCrossentropy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses__binary_focal_crossentropy(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras losses.BinaryFocalCrossentropy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses__categorical_crossentropy(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras losses.CategoricalCrossentropy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses__categorical_focal_crossentropy(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras losses.CategoricalFocalCrossentropy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses__categorical_generalized_cross_entropy(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras losses.CategoricalGeneralizedCrossEntropy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses__categorical_hinge(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras losses.CategoricalHinge operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses__circle(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras losses.Circle operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses__cosine_similarity(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras losses.CosineSimilarity operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses__dice(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras losses.Dice operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses__hinge(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras losses.Hinge operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses__huber(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras losses.Huber operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses_kl_divergence(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras losses.KLDivergence operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses__log_cosh(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras losses.LogCosh operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses__mean_absolute_error(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras losses.MeanAbsoluteError operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses__mean_absolute_percentage_error(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras losses.MeanAbsolutePercentageError operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses__mean_squared_error(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras losses.MeanSquaredError operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses__mean_squared_logarithmic_error(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras losses.MeanSquaredLogarithmicError operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses__poisson(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras losses.Poisson operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses__sparse_categorical_crossentropy(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras losses.SparseCategoricalCrossentropy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses__squared_hinge(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras losses.SquaredHinge operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses__tversky(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras losses.Tversky operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses_binary_crossentropy(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras losses.binary_crossentropy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses_binary_focal_crossentropy(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras losses.binary_focal_crossentropy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses_categorical_crossentropy(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras losses.categorical_crossentropy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses_categorical_focal_crossentropy(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras losses.categorical_focal_crossentropy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses_categorical_generalized_cross_entropy(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras losses.categorical_generalized_cross_entropy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses_categorical_hinge(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras losses.categorical_hinge operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses_circle(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras losses.circle operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses_cosine_similarity(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras losses.cosine_similarity operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses_ctc_ext(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras losses.ctc operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses_dice(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras losses.dice operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses_hinge(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras losses.hinge operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses_huber(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras losses.huber operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses_kl_divergence_ext(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras losses.kl_divergence operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses_log_cosh(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras losses.log_cosh operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses_mean_absolute_error(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras losses.mean_absolute_error operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses_mean_absolute_percentage_error(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras losses.mean_absolute_percentage_error operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses_mean_squared_error(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras losses.mean_squared_error operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses_mean_squared_logarithmic_error(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras losses.mean_squared_logarithmic_error operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses_poisson(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras losses.poisson operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses_sparse_categorical_crossentropy(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras losses.sparse_categorical_crossentropy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses_squared_hinge(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras losses.squared_hinge operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_losses_tversky(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras losses.tversky operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_constraints_deserialize(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras constraints.deserialize operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_constraints_get(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras constraints.get operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_constraints_serialize(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras constraints.serialize operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_constraints__constraint(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras constraints.Constraint operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_constraints__max_norm(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras constraints.MaxNorm operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_constraints_max_norm(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras constraints.max_norm operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_constraints__min_max_norm(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras constraints.MinMaxNorm operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_constraints_min_max_norm(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras constraints.min_max_norm operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_constraints__non_neg(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras constraints.NonNeg operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_constraints_non_neg(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras constraints.non_neg operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_constraints__unit_norm(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras constraints.UnitNorm operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_constraints_unit_norm(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras constraints.unit_norm operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_datasets_cifar10_load_data(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras datasets.cifar10.load_data operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_datasets_fashion_mnist_load_data(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras datasets.fashion_mnist.load_data operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_datasets_boston_housing_load_data(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras datasets.boston_housing.load_data operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_datasets_california_housing_load_data(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras datasets.california_housing.load_data operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_datasets_cifar100_load_data(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras datasets.cifar100.load_data operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_datasets_mnist_load_data(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras datasets.mnist.load_data operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_datasets_imdb_get_word_index(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras datasets.imdb.get_word_index operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_datasets_imdb_load_data(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras datasets.imdb.load_data operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_datasets_reuters_get_label_names(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras datasets.reuters.get_label_names operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_datasets_reuters_get_word_index(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras datasets.reuters.get_word_index operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_datasets_reuters_load_data(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras datasets.reuters.load_data operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_dtype_policies_deserialize(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras dtype_policies.deserialize operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_dtype_policies_get(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras dtype_policies.get operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_dtype_policies_serialize(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras dtype_policies.serialize operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_dtype_policies_d_type_policy(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras dtype_policies.DTypePolicy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_dtype_policies__float_d_type_policy(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras dtype_policies.FloatDTypePolicy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_dtype_policies_gptqd_type_policy(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras dtype_policies.GPTQDTypePolicy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_dtype_policies__quantized_d_type_policy(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras dtype_policies.QuantizedDTypePolicy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_dtype_policies__quantized_float8_d_type_policy(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras dtype_policies.QuantizedFloat8DTypePolicy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_dtype_policies_d_type_policy_map(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras dtype_policies.DTypePolicyMap operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_mixed_precision_d_type_policy(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras mixed_precision.DTypePolicy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_mixed_precision__policy(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras mixed_precision.Policy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_mixed_precision_dtype_policy(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras mixed_precision.dtype_policy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_mixed_precision_global_policy(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras mixed_precision.global_policy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_mixed_precision_set_dtype_policy(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras mixed_precision.set_dtype_policy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_mixed_precision_set_global_policy(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras mixed_precision.set_global_policy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_mixed_precision__loss_scale_optimizer(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras mixed_precision.LossScaleOptimizer operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_random_beta(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras random.beta operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_random_binomial(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras random.binomial operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_random_categorical(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras random.categorical operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_random_dropout(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras random.dropout operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_random_gamma(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras random.gamma operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_random_normal(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras random.normal operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_random_randint(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras random.randint operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_random_shuffle(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras random.shuffle operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_random_truncated_normal(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras random.truncated_normal operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_random_uniform(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras random.uniform operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_random__seed_generator(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras random.SeedGenerator operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_config_disable_flash_attention(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras config.disable_flash_attention operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_config_enable_flash_attention(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras config.enable_flash_attention operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_config_epsilon(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras config.epsilon operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_config_floatx(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras config.floatx operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_config_image_data_format(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras config.image_data_format operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_config_is_flash_attention_enabled(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras config.is_flash_attention_enabled operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_config_is_nnx_enabled(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras config.is_nnx_enabled operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_config_max_epochs(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras config.max_epochs operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_config_max_steps_per_epoch(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras config.max_steps_per_epoch operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_config_set_epsilon(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras config.set_epsilon operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_config_set_floatx(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras config.set_floatx operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_config_set_image_data_format(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras config.set_image_data_format operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_config_set_max_epochs(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras config.set_max_epochs operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_config_set_max_steps_per_epoch(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras config.set_max_steps_per_epoch operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_config_dtype_policy(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras config.dtype_policy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_config_set_dtype_policy(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras config.set_dtype_policy operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_config_enable_unsafe_deserialization(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras config.enable_unsafe_deserialization operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_config_set_backend(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras config.set_backend operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_config_disable_interactive_logging(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras config.disable_interactive_logging operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_config_enable_interactive_logging(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras config.enable_interactive_logging operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_config_is_interactive_logging_enabled(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras config.is_interactive_logging_enabled operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_config_disable_traceback_filtering(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras config.disable_traceback_filtering operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_config_enable_traceback_filtering(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras config.enable_traceback_filtering operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_config_is_traceback_filtering_enabled(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras config.is_traceback_filtering_enabled operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_distribution__data_parallel(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras distribution.DataParallel operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_distribution__device_mesh(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras distribution.DeviceMesh operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_distribution__layout_map(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras distribution.LayoutMap operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_distribution__model_parallel(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras distribution.ModelParallel operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_distribution__tensor_layout(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras distribution.TensorLayout operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_distribution_distribute_tensor(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras distribution.distribute_tensor operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_distribution_distribution(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras distribution.distribution operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_distribution_get_device_count(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras distribution.get_device_count operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_distribution_initialize(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras distribution.initialize operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_distribution_list_devices(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras distribution.list_devices operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_distribution_set_distribution(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras distribution.set_distribution operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_visualization_draw_bounding_boxes(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras visualization.draw_bounding_boxes operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_visualization_draw_segmentation_masks(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras visualization.draw_segmentation_masks operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_visualization_plot_bounding_box_gallery(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras visualization.plot_bounding_box_gallery operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_visualization_plot_image_gallery(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras visualization.plot_image_gallery operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_visualization_plot_segmentation_mask_gallery(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras visualization.plot_segmentation_mask_gallery operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_wrappers_sk_learn_classifier(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras wrappers.SKLearnClassifier operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_wrappers_sk_learn_regressor(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras wrappers.SKLearnRegressor operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_wrappers_sk_learn_transformer(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras wrappers.SKLearnTransformer operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_regularizers_deserialize(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras regularizers.deserialize operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_regularizers_get(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras regularizers.get operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_regularizers_serialize(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras regularizers.serialize operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_regularizers_l1(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras regularizers.L1 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_regularizers_l1_ext(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras regularizers.l1 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_regularizers_l1_l2(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras regularizers.L1L2 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_regularizers_l1_l2_ext(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras regularizers.l1_l2 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_regularizers_l2(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras regularizers.L2 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_regularizers_l2_ext(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras regularizers.l2 operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_regularizers__orthogonal_regularizer(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras regularizers.OrthogonalRegularizer operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_regularizers_orthogonal_regularizer(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras regularizers.orthogonal_regularizer operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_regularizers__regularizer(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras regularizers.Regularizer operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_callbacks__backup_and_restore(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras callbacks.BackupAndRestore operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_callbacks__callback(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras callbacks.Callback operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_callbacks__callback_list(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras callbacks.CallbackList operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_callbacks_csv_logger(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras callbacks.CSVLogger operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_callbacks__early_stopping(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras callbacks.EarlyStopping operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_callbacks__history(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras callbacks.History operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_callbacks__lambda_callback(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras callbacks.LambdaCallback operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_callbacks__learning_rate_scheduler(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras callbacks.LearningRateScheduler operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_callbacks__model_checkpoint(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras callbacks.ModelCheckpoint operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_callbacks__progbar_logger(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras callbacks.ProgbarLogger operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_callbacks__reduce_lr_on_plateau(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras callbacks.ReduceLROnPlateau operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_callbacks__remote_monitor(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras callbacks.RemoteMonitor operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_callbacks__swap_ema_weights(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras callbacks.SwapEMAWeights operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_callbacks__tensor_board(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras callbacks.TensorBoard operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_callbacks__terminate_on_na_n(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras callbacks.TerminateOnNaN operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_optimizers_deserialize(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras optimizers.deserialize operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_optimizers_get(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras optimizers.get operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_optimizers_serialize(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras optimizers.serialize operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_optimizers__adadelta(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras optimizers.Adadelta operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_optimizers__adafactor(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras optimizers.Adafactor operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_optimizers__adagrad(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras optimizers.Adagrad operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_optimizers__adam(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras optimizers.Adam operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_optimizers__adamax(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras optimizers.Adamax operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_optimizers__adam_w(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras optimizers.AdamW operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_optimizers__ftrl(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras optimizers.Ftrl operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_optimizers__lamb(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras optimizers.Lamb operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_optimizers__lion(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras optimizers.Lion operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_optimizers__loss_scale_optimizer(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras optimizers.LossScaleOptimizer operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_optimizers__muon(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras optimizers.Muon operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_optimizers__nadam(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras optimizers.Nadam operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_optimizers__optimizer(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras optimizers.Optimizer operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_optimizers_rm_sprop(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras optimizers.RMSprop operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_optimizers_sgd(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras optimizers.SGD operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_optimizers_schedules__cosine_decay(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras optimizers.schedules.CosineDecay operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_optimizers_schedules__cosine_decay_restarts(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras optimizers.schedules.CosineDecayRestarts operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_optimizers_schedules__exponential_decay(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras optimizers.schedules.ExponentialDecay operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_optimizers_schedules__inverse_time_decay(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras optimizers.schedules.InverseTimeDecay operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_optimizers_schedules__learning_rate_schedule(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras optimizers.schedules.LearningRateSchedule operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_optimizers_schedules__piecewise_constant_decay(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras optimizers.schedules.PiecewiseConstantDecay operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_optimizers_schedules__polynomial_decay(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras optimizers.schedules.PolynomialDecay operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_optimizers_schedules_deserialize(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras optimizers.schedules.deserialize operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_optimizers_schedules_serialize(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras optimizers.schedules.serialize operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers_tfsm_layer(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.TFSMLayer operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers_deserialize(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.deserialize operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers_serialize(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.serialize operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__activation(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Activation operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers_elu(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.ELU operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__leaky_re_lu(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.LeakyReLU operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers_p_re_lu(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.PReLU operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__re_lu(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.ReLU operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__softmax(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Softmax operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__additive_attention(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.AdditiveAttention operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__attention(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Attention operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__group_query_attention(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras layers.GroupQueryAttention operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__multi_head_attention(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras layers.MultiHeadAttention operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__conv1_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Conv1D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__convolution1_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Convolution1D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__conv1_d_transpose(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Conv1DTranspose operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__convolution1_d_transpose(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras layers.Convolution1DTranspose operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__conv2_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Conv2D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__convolution2_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Convolution2D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__conv2_d_transpose(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Conv2DTranspose operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__convolution2_d_transpose(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras layers.Convolution2DTranspose operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__conv3_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Conv3D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__convolution3_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Convolution3D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__conv3_d_transpose(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Conv3DTranspose operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__convolution3_d_transpose(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras layers.Convolution3DTranspose operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__depthwise_conv1_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.DepthwiseConv1D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__depthwise_conv2_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.DepthwiseConv2D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__separable_conv1_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.SeparableConv1D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__separable_convolution1_d(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras layers.SeparableConvolution1D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__separable_conv2_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.SeparableConv2D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__separable_convolution2_d(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras layers.SeparableConvolution2D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__dense(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Dense operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__einsum_dense(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.EinsumDense operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__embedding(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Embedding operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__identity(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Identity operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__input(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Input operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__input_layer(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.InputLayer operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__lambda(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Lambda operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__masking(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Masking operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__reversible_embedding(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras layers.ReversibleEmbedding operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__wrapper(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Wrapper operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__input_spec(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.InputSpec operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__layer(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Layer operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__add(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Add operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers_add(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.add operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__average(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Average operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers_average(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.average operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__concatenate(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Concatenate operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers_concatenate(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.concatenate operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__dot(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Dot operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers_dot(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.dot operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__maximum(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Maximum operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers_maximum(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.maximum operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__minimum(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Minimum operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers_minimum(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.minimum operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__multiply(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Multiply operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers_multiply(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.multiply operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__subtract(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Subtract operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers_subtract(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.subtract operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__batch_normalization(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras layers.BatchNormalization operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__group_normalization(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras layers.GroupNormalization operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__layer_normalization(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras layers.LayerNormalization operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers_rms_normalization(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.RMSNormalization operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__spectral_normalization(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras layers.SpectralNormalization operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__unit_normalization(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.UnitNormalization operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__adaptive_average_pooling1_d(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras layers.AdaptiveAveragePooling1D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__adaptive_average_pooling2_d(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras layers.AdaptiveAveragePooling2D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__adaptive_average_pooling3_d(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras layers.AdaptiveAveragePooling3D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__adaptive_max_pooling1_d(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras layers.AdaptiveMaxPooling1D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__adaptive_max_pooling2_d(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras layers.AdaptiveMaxPooling2D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__adaptive_max_pooling3_d(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras layers.AdaptiveMaxPooling3D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__average_pooling1_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.AveragePooling1D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__avg_pool1_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.AvgPool1D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__average_pooling2_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.AveragePooling2D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__avg_pool2_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.AvgPool2D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__average_pooling3_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.AveragePooling3D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__avg_pool3_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.AvgPool3D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__global_average_pooling1_d(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras layers.GlobalAveragePooling1D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__global_avg_pool1_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.GlobalAvgPool1D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__global_average_pooling2_d(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras layers.GlobalAveragePooling2D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__global_avg_pool2_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.GlobalAvgPool2D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__global_average_pooling3_d(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras layers.GlobalAveragePooling3D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__global_avg_pool3_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.GlobalAvgPool3D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__global_max_pool1_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.GlobalMaxPool1D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__global_max_pooling1_d(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras layers.GlobalMaxPooling1D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__global_max_pool2_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.GlobalMaxPool2D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__global_max_pooling2_d(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras layers.GlobalMaxPooling2D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__global_max_pool3_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.GlobalMaxPool3D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__global_max_pooling3_d(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras layers.GlobalMaxPooling3D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__max_pool1_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.MaxPool1D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__max_pooling1_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.MaxPooling1D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__max_pool2_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.MaxPool2D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__max_pooling2_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.MaxPooling2D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__max_pool3_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.MaxPool3D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__max_pooling3_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.MaxPooling3D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__category_encoding(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.CategoryEncoding operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__discretization(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Discretization operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__hashed_crossing(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.HashedCrossing operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__hashing(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Hashing operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__aug_mix(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.AugMix operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__auto_contrast(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.AutoContrast operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__center_crop(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.CenterCrop operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__cut_mix(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.CutMix operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__equalization(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Equalization operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__max_num_bounding_boxes(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras layers.MaxNumBoundingBoxes operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__mix_up(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.MixUp operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__rand_augment(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.RandAugment operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__random_brightness(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.RandomBrightness operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__random_color_degeneration(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras layers.RandomColorDegeneration operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__random_color_jitter(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras layers.RandomColorJitter operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__random_contrast(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.RandomContrast operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__random_crop(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.RandomCrop operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__random_elastic_transform(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras layers.RandomElasticTransform operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__random_erasing(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.RandomErasing operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__random_flip(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.RandomFlip operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__random_gaussian_blur(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras layers.RandomGaussianBlur operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__random_grayscale(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.RandomGrayscale operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__random_hue(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.RandomHue operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__random_invert(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.RandomInvert operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__random_perspective(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.RandomPerspective operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__random_posterization(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras layers.RandomPosterization operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__random_rotation(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.RandomRotation operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__random_saturation(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.RandomSaturation operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__random_sharpness(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.RandomSharpness operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__random_shear(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.RandomShear operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__random_translation(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.RandomTranslation operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__random_zoom(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.RandomZoom operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__resizing(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Resizing operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__solarization(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Solarization operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__integer_lookup(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.IntegerLookup operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__mel_spectrogram(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.MelSpectrogram operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__normalization(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Normalization operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__pipeline(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Pipeline operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__rescaling(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Rescaling operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers_stft_spectrogram(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.STFTSpectrogram operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__string_lookup(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.StringLookup operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__text_vectorization(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.TextVectorization operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__activity_regularization(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras layers.ActivityRegularization operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__alpha_dropout(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.AlphaDropout operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__dropout(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Dropout operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__gaussian_dropout(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.GaussianDropout operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__gaussian_noise(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.GaussianNoise operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__spatial_dropout1_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.SpatialDropout1D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__spatial_dropout2_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.SpatialDropout2D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__spatial_dropout3_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.SpatialDropout3D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__cropping1_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Cropping1D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__cropping2_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Cropping2D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__cropping3_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Cropping3D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__flatten(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Flatten operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__permute(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Permute operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__repeat_vector(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.RepeatVector operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__reshape(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Reshape operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__up_sampling1_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.UpSampling1D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__up_sampling2_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.UpSampling2D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__up_sampling3_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.UpSampling3D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__zero_padding1_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.ZeroPadding1D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__zero_padding2_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.ZeroPadding2D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__zero_padding3_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.ZeroPadding3D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__bidirectional(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.Bidirectional operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__conv_lstm1_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.ConvLSTM1D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__conv_lstm2_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.ConvLSTM2D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__conv_lstm3_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.ConvLSTM3D operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers_gru(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.GRU operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers_gru_cell(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.GRUCell operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers_lstm(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.LSTM operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers_lstm_cell(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.LSTMCell operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers_rnn(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.RNN operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__simple_rnn(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.SimpleRNN operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__simple_rnn_cell(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.SimpleRNNCell operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__stacked_rnn_cells(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.StackedRNNCells operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__time_distributed(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.TimeDistributed operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__flax_layer(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.FlaxLayer operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__jax_layer(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras layers.JaxLayer operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__torch_module_wrapper(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Execute the  map keras layers.TorchModuleWrapper operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_tree_map_to_none(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras tree.MAP_TO_NONE operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_tree_assert_same_paths(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras tree.assert_same_paths operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_tree_assert_same_structure(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras tree.assert_same_structure operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_tree_flatten(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras tree.flatten operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_tree_flatten_with_path(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras tree.flatten_with_path operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_tree_is_nested(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras tree.is_nested operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_tree_lists_to_tuples(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras tree.lists_to_tuples operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_tree_map_shape_structure(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras tree.map_shape_structure operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_tree_map_structure(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras tree.map_structure operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_tree_map_structure_up_to(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras tree.map_structure_up_to operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_tree_pack_sequence_as(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras tree.pack_sequence_as operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_tree_traverse(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Execute the  map keras tree.traverse operation."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


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
    "Variable": _map_keras_variable,
    "device": _map_keras_device,
    "name_scope": _map_keras_name_scope,
    "KerasTensor": _map_keras_keras_tensor,
    "RematScope": _map_keras_remat_scope,
    "remat": _map_keras_remat,
    "StatelessScope": _map_keras_stateless_scope,
    "SymbolicScope": _map_keras_symbolic_scope,
    "DTypePolicy": _map_keras_d_type_policy,
    "FloatDTypePolicy": _map_keras_float_d_type_policy,
    "Initializer": _map_keras_initializer,
    "Input": _map_keras_input,
    "InputSpec": _map_keras_input_spec,
    "Layer": _map_keras_layer_ext,
    "Loss": _map_keras_loss,
    "Metric": _map_keras_metric,
    "Model": _map_keras_model,
    "Sequential": _map_keras_sequential,
    "Function": _map_keras_function,
    "Operation": _map_keras_operation,
    "Optimizer": _map_keras_optimizer,
    "Quantizer": _map_keras_quantizer,
    "Regularizer": _map_keras_regularizer,
    "version": _map_keras_version,
    "saving.KerasFileEditor": _map_keras_saving__keras_file_editor,
    "saving.CustomObjectScope": _map_keras_saving__custom_object_scope,
    "saving.custom_object_scope": _map_keras_saving_custom_object_scope,
    "saving.get_custom_objects": _map_keras_saving_get_custom_objects,
    "saving.get_registered_name": _map_keras_saving_get_registered_name,
    "saving.get_registered_object": _map_keras_saving_get_registered_object,
    "saving.register_keras_serializable": _map_keras_saving_register_keras_serializable,
    "saving.load_model": _map_keras_saving_load_model,
    "saving.load_weights": _map_keras_saving_load_weights,
    "saving.save_model": _map_keras_saving_save_model,
    "saving.save_weights": _map_keras_saving_save_weights,
    "saving.deserialize_keras_object": _map_keras_saving_deserialize_keras_object,
    "saving.serialize_keras_object": _map_keras_saving_serialize_keras_object,
    "export.ExportArchive": _map_keras_export__export_archive,
    "preprocessing.image_dataset_from_directory": _map_keras_preprocessing_image_dataset_from_directory,
    "preprocessing.text_dataset_from_directory": _map_keras_preprocessing_text_dataset_from_directory,
    "preprocessing.timeseries_dataset_from_array": _map_keras_preprocessing_timeseries_dataset_from_array,
    "preprocessing.image.array_to_img": _map_keras_preprocessing_image_array_to_img,
    "preprocessing.image.img_to_array": _map_keras_preprocessing_image_img_to_array,
    "preprocessing.image.load_img": _map_keras_preprocessing_image_load_img,
    "preprocessing.image.save_img": _map_keras_preprocessing_image_save_img,
    "preprocessing.image.smart_resize": _map_keras_preprocessing_image_smart_resize,
    "preprocessing.sequence.pad_sequences": _map_keras_preprocessing_sequence_pad_sequences,
    "distillation.DistillationLoss": _map_keras_distillation__distillation_loss,
    "distillation.FeatureDistillation": _map_keras_distillation__feature_distillation,
    "distillation.LogitsDistillation": _map_keras_distillation__logits_distillation,
    "distillation.Distiller": _map_keras_distillation__distiller,
    "models.clone_model": _map_keras_models_clone_model,
    "models.Model": _map_keras_models__model,
    "models.model_from_json": _map_keras_models_model_from_json,
    "models.Sequential": _map_keras_models__sequential,
    "models.load_model": _map_keras_models_load_model,
    "models.save_model": _map_keras_models_save_model,
    "applications.ConvNeXtBase": _map_keras_applications__conv_ne_xt_base,
    "applications.ConvNeXtLarge": _map_keras_applications__conv_ne_xt_large,
    "applications.ConvNeXtSmall": _map_keras_applications__conv_ne_xt_small,
    "applications.ConvNeXtTiny": _map_keras_applications__conv_ne_xt_tiny,
    "applications.ConvNeXtXLarge": _map_keras_applications__conv_ne_xt_x_large,
    "applications.DenseNet121": _map_keras_applications__dense_net121,
    "applications.DenseNet169": _map_keras_applications__dense_net169,
    "applications.DenseNet201": _map_keras_applications__dense_net201,
    "applications.EfficientNetB0": _map_keras_applications__efficient_net_b0,
    "applications.EfficientNetB1": _map_keras_applications__efficient_net_b1,
    "applications.EfficientNetB2": _map_keras_applications__efficient_net_b2,
    "applications.EfficientNetB3": _map_keras_applications__efficient_net_b3,
    "applications.EfficientNetB4": _map_keras_applications__efficient_net_b4,
    "applications.EfficientNetB5": _map_keras_applications__efficient_net_b5,
    "applications.EfficientNetB6": _map_keras_applications__efficient_net_b6,
    "applications.EfficientNetB7": _map_keras_applications__efficient_net_b7,
    "applications.EfficientNetV2B0": _map_keras_applications__efficient_net_v2_b0,
    "applications.EfficientNetV2B1": _map_keras_applications__efficient_net_v2_b1,
    "applications.EfficientNetV2B2": _map_keras_applications__efficient_net_v2_b2,
    "applications.EfficientNetV2B3": _map_keras_applications__efficient_net_v2_b3,
    "applications.EfficientNetV2L": _map_keras_applications__efficient_net_v2_l,
    "applications.EfficientNetV2M": _map_keras_applications__efficient_net_v2_m,
    "applications.EfficientNetV2S": _map_keras_applications__efficient_net_v2_s,
    "applications.InceptionResNetV2": _map_keras_applications__inception_res_net_v2,
    "applications.InceptionV3": _map_keras_applications__inception_v3,
    "applications.MobileNet": _map_keras_applications__mobile_net,
    "applications.MobileNetV2": _map_keras_applications__mobile_net_v2,
    "applications.MobileNetV3Large": _map_keras_applications__mobile_net_v3_large,
    "applications.MobileNetV3Small": _map_keras_applications__mobile_net_v3_small,
    "applications.NASNetLarge": _map_keras_applications_nas_net_large,
    "applications.NASNetMobile": _map_keras_applications_nas_net_mobile,
    "applications.ResNet50": _map_keras_applications__res_net50,
    "applications.ResNet101": _map_keras_applications__res_net101,
    "applications.ResNet152": _map_keras_applications__res_net152,
    "applications.ResNet50V2": _map_keras_applications__res_net50_v2,
    "applications.ResNet101V2": _map_keras_applications__res_net101_v2,
    "applications.ResNet152V2": _map_keras_applications__res_net152_v2,
    "applications.VGG16": _map_keras_applications_vgg16,
    "applications.VGG19": _map_keras_applications_vgg19,
    "applications.Xception": _map_keras_applications__xception,
    "applications.mobilenet_v3.decode_predictions": _map_keras_applications_mobilenet_v3_decode_predictions,
    "applications.mobilenet_v3.preprocess_input": _map_keras_applications_mobilenet_v3_preprocess_input,
    "applications.inception_resnet_v2.InceptionResNetV2": _map_keras_applications_inception_resnet_v2__inception_res_net_v2,
    "applications.inception_resnet_v2.decode_predictions": _map_keras_applications_inception_resnet_v2_decode_predictions,
    "applications.inception_resnet_v2.preprocess_input": _map_keras_applications_inception_resnet_v2_preprocess_input,
    "applications.resnet_v2.ResNet50V2": _map_keras_applications_resnet_v2__res_net50_v2,
    "applications.resnet_v2.ResNet101V2": _map_keras_applications_resnet_v2__res_net101_v2,
    "applications.resnet_v2.ResNet152V2": _map_keras_applications_resnet_v2__res_net152_v2,
    "applications.resnet_v2.decode_predictions": _map_keras_applications_resnet_v2_decode_predictions,
    "applications.resnet_v2.preprocess_input": _map_keras_applications_resnet_v2_preprocess_input,
    "applications.convnext.ConvNeXtBase": _map_keras_applications_convnext__conv_ne_xt_base,
    "applications.convnext.ConvNeXtLarge": _map_keras_applications_convnext__conv_ne_xt_large,
    "applications.convnext.ConvNeXtSmall": _map_keras_applications_convnext__conv_ne_xt_small,
    "applications.convnext.ConvNeXtTiny": _map_keras_applications_convnext__conv_ne_xt_tiny,
    "applications.convnext.ConvNeXtXLarge": _map_keras_applications_convnext__conv_ne_xt_x_large,
    "applications.convnext.decode_predictions": _map_keras_applications_convnext_decode_predictions,
    "applications.convnext.preprocess_input": _map_keras_applications_convnext_preprocess_input,
    "applications.inception_v3.InceptionV3": _map_keras_applications_inception_v3__inception_v3,
    "applications.inception_v3.decode_predictions": _map_keras_applications_inception_v3_decode_predictions,
    "applications.inception_v3.preprocess_input": _map_keras_applications_inception_v3_preprocess_input,
    "applications.xception.Xception": _map_keras_applications_xception__xception,
    "applications.xception.decode_predictions": _map_keras_applications_xception_decode_predictions,
    "applications.xception.preprocess_input": _map_keras_applications_xception_preprocess_input,
    "applications.resnet50.ResNet50": _map_keras_applications_resnet50__res_net50,
    "applications.resnet50.decode_predictions": _map_keras_applications_resnet50_decode_predictions,
    "applications.resnet50.preprocess_input": _map_keras_applications_resnet50_preprocess_input,
    "applications.vgg19.VGG19": _map_keras_applications_vgg19_vgg19,
    "applications.vgg19.decode_predictions": _map_keras_applications_vgg19_decode_predictions,
    "applications.vgg19.preprocess_input": _map_keras_applications_vgg19_preprocess_input,
    "applications.resnet.ResNet50": _map_keras_applications_resnet__res_net50,
    "applications.resnet.ResNet101": _map_keras_applications_resnet__res_net101,
    "applications.resnet.ResNet152": _map_keras_applications_resnet__res_net152,
    "applications.resnet.decode_predictions": _map_keras_applications_resnet_decode_predictions,
    "applications.resnet.preprocess_input": _map_keras_applications_resnet_preprocess_input,
    "applications.vgg16.VGG16": _map_keras_applications_vgg16_vgg16,
    "applications.vgg16.decode_predictions": _map_keras_applications_vgg16_decode_predictions,
    "applications.vgg16.preprocess_input": _map_keras_applications_vgg16_preprocess_input,
    "applications.densenet.DenseNet121": _map_keras_applications_densenet__dense_net121,
    "applications.densenet.DenseNet169": _map_keras_applications_densenet__dense_net169,
    "applications.densenet.DenseNet201": _map_keras_applications_densenet__dense_net201,
    "applications.densenet.decode_predictions": _map_keras_applications_densenet_decode_predictions,
    "applications.densenet.preprocess_input": _map_keras_applications_densenet_preprocess_input,
    "applications.nasnet.NASNetLarge": _map_keras_applications_nasnet_nas_net_large,
    "applications.nasnet.NASNetMobile": _map_keras_applications_nasnet_nas_net_mobile,
    "applications.nasnet.decode_predictions": _map_keras_applications_nasnet_decode_predictions,
    "applications.nasnet.preprocess_input": _map_keras_applications_nasnet_preprocess_input,
    "applications.mobilenet.MobileNet": _map_keras_applications_mobilenet__mobile_net,
    "applications.mobilenet.decode_predictions": _map_keras_applications_mobilenet_decode_predictions,
    "applications.mobilenet.preprocess_input": _map_keras_applications_mobilenet_preprocess_input,
    "applications.imagenet_utils.decode_predictions": _map_keras_applications_imagenet_utils_decode_predictions,
    "applications.imagenet_utils.preprocess_input": _map_keras_applications_imagenet_utils_preprocess_input,
    "applications.efficientnet.EfficientNetB0": _map_keras_applications_efficientnet__efficient_net_b0,
    "applications.efficientnet.EfficientNetB1": _map_keras_applications_efficientnet__efficient_net_b1,
    "applications.efficientnet.EfficientNetB2": _map_keras_applications_efficientnet__efficient_net_b2,
    "applications.efficientnet.EfficientNetB3": _map_keras_applications_efficientnet__efficient_net_b3,
    "applications.efficientnet.EfficientNetB4": _map_keras_applications_efficientnet__efficient_net_b4,
    "applications.efficientnet.EfficientNetB5": _map_keras_applications_efficientnet__efficient_net_b5,
    "applications.efficientnet.EfficientNetB6": _map_keras_applications_efficientnet__efficient_net_b6,
    "applications.efficientnet.EfficientNetB7": _map_keras_applications_efficientnet__efficient_net_b7,
    "applications.efficientnet.decode_predictions": _map_keras_applications_efficientnet_decode_predictions,
    "applications.efficientnet.preprocess_input": _map_keras_applications_efficientnet_preprocess_input,
    "applications.efficientnet_v2.EfficientNetV2B0": _map_keras_applications_efficientnet_v2__efficient_net_v2_b0,
    "applications.efficientnet_v2.EfficientNetV2B1": _map_keras_applications_efficientnet_v2__efficient_net_v2_b1,
    "applications.efficientnet_v2.EfficientNetV2B2": _map_keras_applications_efficientnet_v2__efficient_net_v2_b2,
    "applications.efficientnet_v2.EfficientNetV2B3": _map_keras_applications_efficientnet_v2__efficient_net_v2_b3,
    "applications.efficientnet_v2.EfficientNetV2L": _map_keras_applications_efficientnet_v2__efficient_net_v2_l,
    "applications.efficientnet_v2.EfficientNetV2M": _map_keras_applications_efficientnet_v2__efficient_net_v2_m,
    "applications.efficientnet_v2.EfficientNetV2S": _map_keras_applications_efficientnet_v2__efficient_net_v2_s,
    "applications.efficientnet_v2.decode_predictions": _map_keras_applications_efficientnet_v2_decode_predictions,
    "applications.efficientnet_v2.preprocess_input": _map_keras_applications_efficientnet_v2_preprocess_input,
    "applications.mobilenet_v2.MobileNetV2": _map_keras_applications_mobilenet_v2__mobile_net_v2,
    "applications.mobilenet_v2.decode_predictions": _map_keras_applications_mobilenet_v2_decode_predictions,
    "applications.mobilenet_v2.preprocess_input": _map_keras_applications_mobilenet_v2_preprocess_input,
    "initializers.deserialize": _map_keras_initializers_deserialize,
    "initializers.get": _map_keras_initializers_get,
    "initializers.serialize": _map_keras_initializers_serialize,
    "initializers.STFT": _map_keras_initializers_stft,
    "initializers.STFTInitializer": _map_keras_initializers_stft_initializer,
    "initializers.stft": _map_keras_initializers_stft_ext,
    "initializers.Constant": _map_keras_initializers__constant,
    "initializers.constant": _map_keras_initializers_constant,
    "initializers.Identity": _map_keras_initializers__identity,
    "initializers.IdentityInitializer": _map_keras_initializers__identity_initializer,
    "initializers.identity": _map_keras_initializers_identity,
    "initializers.Ones": _map_keras_initializers__ones,
    "initializers.ones": _map_keras_initializers_ones,
    "initializers.Zeros": _map_keras_initializers__zeros,
    "initializers.zeros": _map_keras_initializers_zeros,
    "initializers.Initializer": _map_keras_initializers__initializer,
    "initializers.GlorotNormal": _map_keras_initializers__glorot_normal,
    "initializers.glorot_normal": _map_keras_initializers_glorot_normal,
    "initializers.GlorotUniform": _map_keras_initializers__glorot_uniform,
    "initializers.glorot_uniform": _map_keras_initializers_glorot_uniform,
    "initializers.HeNormal": _map_keras_initializers__he_normal,
    "initializers.he_normal": _map_keras_initializers_he_normal,
    "initializers.HeUniform": _map_keras_initializers__he_uniform,
    "initializers.he_uniform": _map_keras_initializers_he_uniform,
    "initializers.LecunNormal": _map_keras_initializers__lecun_normal,
    "initializers.lecun_normal": _map_keras_initializers_lecun_normal,
    "initializers.LecunUniform": _map_keras_initializers__lecun_uniform,
    "initializers.lecun_uniform": _map_keras_initializers_lecun_uniform,
    "initializers.Orthogonal": _map_keras_initializers__orthogonal,
    "initializers.OrthogonalInitializer": _map_keras_initializers__orthogonal_initializer,
    "initializers.orthogonal": _map_keras_initializers_orthogonal,
    "initializers.RandomNormal": _map_keras_initializers__random_normal,
    "initializers.random_normal": _map_keras_initializers_random_normal,
    "initializers.RandomUniform": _map_keras_initializers__random_uniform,
    "initializers.random_uniform": _map_keras_initializers_random_uniform,
    "initializers.TruncatedNormal": _map_keras_initializers__truncated_normal,
    "initializers.truncated_normal": _map_keras_initializers_truncated_normal,
    "initializers.VarianceScaling": _map_keras_initializers__variance_scaling,
    "initializers.variance_scaling": _map_keras_initializers_variance_scaling,
    "quantizers.deserialize": _map_keras_quantizers_deserialize,
    "quantizers.get": _map_keras_quantizers_get,
    "quantizers.serialize": _map_keras_quantizers_serialize,
    "quantizers.GPTQConfig": _map_keras_quantizers_gptq_config,
    "quantizers.Float8QuantizationConfig": _map_keras_quantizers__float8_quantization_config,
    "quantizers.Int4QuantizationConfig": _map_keras_quantizers__int4_quantization_config,
    "quantizers.Int8QuantizationConfig": _map_keras_quantizers__int8_quantization_config,
    "quantizers.QuantizationConfig": _map_keras_quantizers__quantization_config,
    "quantizers.AbsMaxQuantizer": _map_keras_quantizers__abs_max_quantizer,
    "quantizers.Quantizer": _map_keras_quantizers__quantizer,
    "quantizers.abs_max_quantize": _map_keras_quantizers_abs_max_quantize,
    "quantizers.compute_float8_amax_history": _map_keras_quantizers_compute_float8_amax_history,
    "quantizers.compute_float8_scale": _map_keras_quantizers_compute_float8_scale,
    "quantizers.fake_quant_with_min_max_vars": _map_keras_quantizers_fake_quant_with_min_max_vars,
    "quantizers.pack_int4": _map_keras_quantizers_pack_int4,
    "quantizers.quantize_and_dequantize": _map_keras_quantizers_quantize_and_dequantize,
    "quantizers.unpack_int4": _map_keras_quantizers_unpack_int4,
    "activations.deserialize": _map_keras_activations_deserialize,
    "activations.get": _map_keras_activations_get,
    "activations.serialize": _map_keras_activations_serialize,
    "activations.celu": _map_keras_activations_celu,
    "activations.elu": _map_keras_activations_elu,
    "activations.exponential": _map_keras_activations_exponential,
    "activations.gelu": _map_keras_activations_gelu,
    "activations.glu": _map_keras_activations_glu,
    "activations.hard_shrink": _map_keras_activations_hard_shrink,
    "activations.hard_sigmoid": _map_keras_activations_hard_sigmoid,
    "activations.hard_silu": _map_keras_activations_hard_silu,
    "activations.hard_swish": _map_keras_activations_hard_swish,
    "activations.hard_tanh": _map_keras_activations_hard_tanh,
    "activations.leaky_relu": _map_keras_activations_leaky_relu,
    "activations.linear": _map_keras_activations_linear,
    "activations.log_sigmoid": _map_keras_activations_log_sigmoid,
    "activations.log_softmax": _map_keras_activations_log_softmax,
    "activations.mish": _map_keras_activations_mish,
    "activations.relu": _map_keras_activations_relu,
    "activations.relu6": _map_keras_activations_relu6,
    "activations.selu": _map_keras_activations_selu,
    "activations.sigmoid": _map_keras_activations_sigmoid,
    "activations.silu": _map_keras_activations_silu,
    "activations.swish": _map_keras_activations_swish,
    "activations.soft_shrink": _map_keras_activations_soft_shrink,
    "activations.softmax": _map_keras_activations_softmax,
    "activations.softplus": _map_keras_activations_softplus,
    "activations.softsign": _map_keras_activations_softsign,
    "activations.sparse_plus": _map_keras_activations_sparse_plus,
    "activations.sparse_sigmoid": _map_keras_activations_sparse_sigmoid,
    "activations.sparsemax": _map_keras_activations_sparsemax,
    "activations.squareplus": _map_keras_activations_squareplus,
    "activations.tanh": _map_keras_activations_tanh,
    "activations.tanh_shrink": _map_keras_activations_tanh_shrink,
    "activations.threshold": _map_keras_activations_threshold,
    "metrics.binary_crossentropy": _map_keras_metrics_binary_crossentropy,
    "metrics.binary_focal_crossentropy": _map_keras_metrics_binary_focal_crossentropy,
    "metrics.categorical_crossentropy": _map_keras_metrics_categorical_crossentropy,
    "metrics.categorical_focal_crossentropy": _map_keras_metrics_categorical_focal_crossentropy,
    "metrics.categorical_hinge": _map_keras_metrics_categorical_hinge,
    "metrics.hinge": _map_keras_metrics_hinge,
    "metrics.huber": _map_keras_metrics_huber,
    "metrics.kl_divergence": _map_keras_metrics_kl_divergence,
    "metrics.log_cosh": _map_keras_metrics_log_cosh,
    "metrics.mean_absolute_error": _map_keras_metrics_mean_absolute_error,
    "metrics.mean_absolute_percentage_error": _map_keras_metrics_mean_absolute_percentage_error,
    "metrics.mean_squared_error": _map_keras_metrics_mean_squared_error,
    "metrics.mean_squared_logarithmic_error": _map_keras_metrics_mean_squared_logarithmic_error,
    "metrics.poisson": _map_keras_metrics_poisson,
    "metrics.sparse_categorical_crossentropy": _map_keras_metrics_sparse_categorical_crossentropy,
    "metrics.squared_hinge": _map_keras_metrics_squared_hinge,
    "metrics.deserialize": _map_keras_metrics_deserialize,
    "metrics.get": _map_keras_metrics_get,
    "metrics.serialize": _map_keras_metrics_serialize,
    "metrics.Accuracy": _map_keras_metrics__accuracy,
    "metrics.BinaryAccuracy": _map_keras_metrics__binary_accuracy,
    "metrics.CategoricalAccuracy": _map_keras_metrics__categorical_accuracy,
    "metrics.SparseCategoricalAccuracy": _map_keras_metrics__sparse_categorical_accuracy,
    "metrics.SparseTopKCategoricalAccuracy": _map_keras_metrics__sparse_top_k_categorical_accuracy,
    "metrics.TopKCategoricalAccuracy": _map_keras_metrics__top_k_categorical_accuracy,
    "metrics.binary_accuracy": _map_keras_metrics_binary_accuracy,
    "metrics.categorical_accuracy": _map_keras_metrics_categorical_accuracy,
    "metrics.sparse_categorical_accuracy": _map_keras_metrics_sparse_categorical_accuracy,
    "metrics.sparse_top_k_categorical_accuracy": _map_keras_metrics_sparse_top_k_categorical_accuracy,
    "metrics.top_k_categorical_accuracy": _map_keras_metrics_top_k_categorical_accuracy,
    "metrics.AUC": _map_keras_metrics_auc,
    "metrics.FalseNegatives": _map_keras_metrics__false_negatives,
    "metrics.FalsePositives": _map_keras_metrics__false_positives,
    "metrics.Precision": _map_keras_metrics__precision,
    "metrics.PrecisionAtRecall": _map_keras_metrics__precision_at_recall,
    "metrics.Recall": _map_keras_metrics__recall,
    "metrics.RecallAtPrecision": _map_keras_metrics__recall_at_precision,
    "metrics.SensitivityAtSpecificity": _map_keras_metrics__sensitivity_at_specificity,
    "metrics.SpecificityAtSensitivity": _map_keras_metrics__specificity_at_sensitivity,
    "metrics.TrueNegatives": _map_keras_metrics__true_negatives,
    "metrics.TruePositives": _map_keras_metrics__true_positives,
    "metrics.ConcordanceCorrelation": _map_keras_metrics__concordance_correlation,
    "metrics.PearsonCorrelation": _map_keras_metrics__pearson_correlation,
    "metrics.concordance_correlation": _map_keras_metrics_concordance_correlation,
    "metrics.pearson_correlation": _map_keras_metrics_pearson_correlation,
    "metrics.F1Score": _map_keras_metrics_f1_score,
    "metrics.FBetaScore": _map_keras_metrics_f_beta_score,
    "metrics.CategoricalHinge": _map_keras_metrics__categorical_hinge,
    "metrics.Hinge": _map_keras_metrics__hinge,
    "metrics.SquaredHinge": _map_keras_metrics__squared_hinge,
    "metrics.BinaryIoU": _map_keras_metrics__binary_io_u,
    "metrics.IoU": _map_keras_metrics__io_u,
    "metrics.MeanIoU": _map_keras_metrics__mean_io_u,
    "metrics.OneHotIoU": _map_keras_metrics__one_hot_io_u,
    "metrics.OneHotMeanIoU": _map_keras_metrics__one_hot_mean_io_u,
    "metrics.Metric": _map_keras_metrics__metric,
    "metrics.BinaryCrossentropy": _map_keras_metrics__binary_crossentropy,
    "metrics.CategoricalCrossentropy": _map_keras_metrics__categorical_crossentropy,
    "metrics.KLDivergence": _map_keras_metrics_kl_divergence_ext,
    "metrics.Poisson": _map_keras_metrics__poisson,
    "metrics.SparseCategoricalCrossentropy": _map_keras_metrics__sparse_categorical_crossentropy,
    "metrics.Mean": _map_keras_metrics__mean,
    "metrics.MeanMetricWrapper": _map_keras_metrics__mean_metric_wrapper,
    "metrics.Sum": _map_keras_metrics__sum,
    "metrics.CosineSimilarity": _map_keras_metrics__cosine_similarity,
    "metrics.LogCoshError": _map_keras_metrics__log_cosh_error,
    "metrics.MeanAbsoluteError": _map_keras_metrics__mean_absolute_error,
    "metrics.MeanAbsolutePercentageError": _map_keras_metrics__mean_absolute_percentage_error,
    "metrics.MeanSquaredError": _map_keras_metrics__mean_squared_error,
    "metrics.MeanSquaredLogarithmicError": _map_keras_metrics__mean_squared_logarithmic_error,
    "metrics.R2Score": _map_keras_metrics_r2_score,
    "metrics.RootMeanSquaredError": _map_keras_metrics__root_mean_squared_error,
    "losses.deserialize": _map_keras_losses_deserialize,
    "losses.get": _map_keras_losses_get,
    "losses.serialize": _map_keras_losses_serialize,
    "losses.Loss": _map_keras_losses__loss,
    "losses.CTC": _map_keras_losses_ctc,
    "losses.BinaryCrossentropy": _map_keras_losses__binary_crossentropy,
    "losses.BinaryFocalCrossentropy": _map_keras_losses__binary_focal_crossentropy,
    "losses.CategoricalCrossentropy": _map_keras_losses__categorical_crossentropy,
    "losses.CategoricalFocalCrossentropy": _map_keras_losses__categorical_focal_crossentropy,
    "losses.CategoricalGeneralizedCrossEntropy": _map_keras_losses__categorical_generalized_cross_entropy,
    "losses.CategoricalHinge": _map_keras_losses__categorical_hinge,
    "losses.Circle": _map_keras_losses__circle,
    "losses.CosineSimilarity": _map_keras_losses__cosine_similarity,
    "losses.Dice": _map_keras_losses__dice,
    "losses.Hinge": _map_keras_losses__hinge,
    "losses.Huber": _map_keras_losses__huber,
    "losses.KLDivergence": _map_keras_losses_kl_divergence,
    "losses.LogCosh": _map_keras_losses__log_cosh,
    "losses.MeanAbsoluteError": _map_keras_losses__mean_absolute_error,
    "losses.MeanAbsolutePercentageError": _map_keras_losses__mean_absolute_percentage_error,
    "losses.MeanSquaredError": _map_keras_losses__mean_squared_error,
    "losses.MeanSquaredLogarithmicError": _map_keras_losses__mean_squared_logarithmic_error,
    "losses.Poisson": _map_keras_losses__poisson,
    "losses.SparseCategoricalCrossentropy": _map_keras_losses__sparse_categorical_crossentropy,
    "losses.SquaredHinge": _map_keras_losses__squared_hinge,
    "losses.Tversky": _map_keras_losses__tversky,
    "losses.binary_crossentropy": _map_keras_losses_binary_crossentropy,
    "losses.binary_focal_crossentropy": _map_keras_losses_binary_focal_crossentropy,
    "losses.categorical_crossentropy": _map_keras_losses_categorical_crossentropy,
    "losses.categorical_focal_crossentropy": _map_keras_losses_categorical_focal_crossentropy,
    "losses.categorical_generalized_cross_entropy": _map_keras_losses_categorical_generalized_cross_entropy,
    "losses.categorical_hinge": _map_keras_losses_categorical_hinge,
    "losses.circle": _map_keras_losses_circle,
    "losses.cosine_similarity": _map_keras_losses_cosine_similarity,
    "losses.ctc": _map_keras_losses_ctc_ext,
    "losses.dice": _map_keras_losses_dice,
    "losses.hinge": _map_keras_losses_hinge,
    "losses.huber": _map_keras_losses_huber,
    "losses.kl_divergence": _map_keras_losses_kl_divergence_ext,
    "losses.log_cosh": _map_keras_losses_log_cosh,
    "losses.mean_absolute_error": _map_keras_losses_mean_absolute_error,
    "losses.mean_absolute_percentage_error": _map_keras_losses_mean_absolute_percentage_error,
    "losses.mean_squared_error": _map_keras_losses_mean_squared_error,
    "losses.mean_squared_logarithmic_error": _map_keras_losses_mean_squared_logarithmic_error,
    "losses.poisson": _map_keras_losses_poisson,
    "losses.sparse_categorical_crossentropy": _map_keras_losses_sparse_categorical_crossentropy,
    "losses.squared_hinge": _map_keras_losses_squared_hinge,
    "losses.tversky": _map_keras_losses_tversky,
    "constraints.deserialize": _map_keras_constraints_deserialize,
    "constraints.get": _map_keras_constraints_get,
    "constraints.serialize": _map_keras_constraints_serialize,
    "constraints.Constraint": _map_keras_constraints__constraint,
    "constraints.MaxNorm": _map_keras_constraints__max_norm,
    "constraints.max_norm": _map_keras_constraints_max_norm,
    "constraints.MinMaxNorm": _map_keras_constraints__min_max_norm,
    "constraints.min_max_norm": _map_keras_constraints_min_max_norm,
    "constraints.NonNeg": _map_keras_constraints__non_neg,
    "constraints.non_neg": _map_keras_constraints_non_neg,
    "constraints.UnitNorm": _map_keras_constraints__unit_norm,
    "constraints.unit_norm": _map_keras_constraints_unit_norm,
    "datasets.cifar10.load_data": _map_keras_datasets_cifar10_load_data,
    "datasets.fashion_mnist.load_data": _map_keras_datasets_fashion_mnist_load_data,
    "datasets.boston_housing.load_data": _map_keras_datasets_boston_housing_load_data,
    "datasets.california_housing.load_data": _map_keras_datasets_california_housing_load_data,
    "datasets.cifar100.load_data": _map_keras_datasets_cifar100_load_data,
    "datasets.mnist.load_data": _map_keras_datasets_mnist_load_data,
    "datasets.imdb.get_word_index": _map_keras_datasets_imdb_get_word_index,
    "datasets.imdb.load_data": _map_keras_datasets_imdb_load_data,
    "datasets.reuters.get_label_names": _map_keras_datasets_reuters_get_label_names,
    "datasets.reuters.get_word_index": _map_keras_datasets_reuters_get_word_index,
    "datasets.reuters.load_data": _map_keras_datasets_reuters_load_data,
    "dtype_policies.deserialize": _map_keras_dtype_policies_deserialize,
    "dtype_policies.get": _map_keras_dtype_policies_get,
    "dtype_policies.serialize": _map_keras_dtype_policies_serialize,
    "dtype_policies.DTypePolicy": _map_keras_dtype_policies_d_type_policy,
    "dtype_policies.FloatDTypePolicy": _map_keras_dtype_policies__float_d_type_policy,
    "dtype_policies.GPTQDTypePolicy": _map_keras_dtype_policies_gptqd_type_policy,
    "dtype_policies.QuantizedDTypePolicy": _map_keras_dtype_policies__quantized_d_type_policy,
    "dtype_policies.QuantizedFloat8DTypePolicy": _map_keras_dtype_policies__quantized_float8_d_type_policy,
    "dtype_policies.DTypePolicyMap": _map_keras_dtype_policies_d_type_policy_map,
    "mixed_precision.DTypePolicy": _map_keras_mixed_precision_d_type_policy,
    "mixed_precision.Policy": _map_keras_mixed_precision__policy,
    "mixed_precision.dtype_policy": _map_keras_mixed_precision_dtype_policy,
    "mixed_precision.global_policy": _map_keras_mixed_precision_global_policy,
    "mixed_precision.set_dtype_policy": _map_keras_mixed_precision_set_dtype_policy,
    "mixed_precision.set_global_policy": _map_keras_mixed_precision_set_global_policy,
    "mixed_precision.LossScaleOptimizer": _map_keras_mixed_precision__loss_scale_optimizer,
    "random.beta": _map_keras_random_beta,
    "random.binomial": _map_keras_random_binomial,
    "random.categorical": _map_keras_random_categorical,
    "random.dropout": _map_keras_random_dropout,
    "random.gamma": _map_keras_random_gamma,
    "random.normal": _map_keras_random_normal,
    "random.randint": _map_keras_random_randint,
    "random.shuffle": _map_keras_random_shuffle,
    "random.truncated_normal": _map_keras_random_truncated_normal,
    "random.uniform": _map_keras_random_uniform,
    "random.SeedGenerator": _map_keras_random__seed_generator,
    "config.disable_flash_attention": _map_keras_config_disable_flash_attention,
    "config.enable_flash_attention": _map_keras_config_enable_flash_attention,
    "config.epsilon": _map_keras_config_epsilon,
    "config.floatx": _map_keras_config_floatx,
    "config.image_data_format": _map_keras_config_image_data_format,
    "config.is_flash_attention_enabled": _map_keras_config_is_flash_attention_enabled,
    "config.is_nnx_enabled": _map_keras_config_is_nnx_enabled,
    "config.max_epochs": _map_keras_config_max_epochs,
    "config.max_steps_per_epoch": _map_keras_config_max_steps_per_epoch,
    "config.set_epsilon": _map_keras_config_set_epsilon,
    "config.set_floatx": _map_keras_config_set_floatx,
    "config.set_image_data_format": _map_keras_config_set_image_data_format,
    "config.set_max_epochs": _map_keras_config_set_max_epochs,
    "config.set_max_steps_per_epoch": _map_keras_config_set_max_steps_per_epoch,
    "config.dtype_policy": _map_keras_config_dtype_policy,
    "config.set_dtype_policy": _map_keras_config_set_dtype_policy,
    "config.enable_unsafe_deserialization": _map_keras_config_enable_unsafe_deserialization,
    "config.set_backend": _map_keras_config_set_backend,
    "config.disable_interactive_logging": _map_keras_config_disable_interactive_logging,
    "config.enable_interactive_logging": _map_keras_config_enable_interactive_logging,
    "config.is_interactive_logging_enabled": _map_keras_config_is_interactive_logging_enabled,
    "config.disable_traceback_filtering": _map_keras_config_disable_traceback_filtering,
    "config.enable_traceback_filtering": _map_keras_config_enable_traceback_filtering,
    "config.is_traceback_filtering_enabled": _map_keras_config_is_traceback_filtering_enabled,
    "distribution.DataParallel": _map_keras_distribution__data_parallel,
    "distribution.DeviceMesh": _map_keras_distribution__device_mesh,
    "distribution.LayoutMap": _map_keras_distribution__layout_map,
    "distribution.ModelParallel": _map_keras_distribution__model_parallel,
    "distribution.TensorLayout": _map_keras_distribution__tensor_layout,
    "distribution.distribute_tensor": _map_keras_distribution_distribute_tensor,
    "distribution.distribution": _map_keras_distribution_distribution,
    "distribution.get_device_count": _map_keras_distribution_get_device_count,
    "distribution.initialize": _map_keras_distribution_initialize,
    "distribution.list_devices": _map_keras_distribution_list_devices,
    "distribution.set_distribution": _map_keras_distribution_set_distribution,
    "visualization.draw_bounding_boxes": _map_keras_visualization_draw_bounding_boxes,
    "visualization.draw_segmentation_masks": _map_keras_visualization_draw_segmentation_masks,
    "visualization.plot_bounding_box_gallery": _map_keras_visualization_plot_bounding_box_gallery,
    "visualization.plot_image_gallery": _map_keras_visualization_plot_image_gallery,
    "visualization.plot_segmentation_mask_gallery": _map_keras_visualization_plot_segmentation_mask_gallery,
    "wrappers.SKLearnClassifier": _map_keras_wrappers_sk_learn_classifier,
    "wrappers.SKLearnRegressor": _map_keras_wrappers_sk_learn_regressor,
    "wrappers.SKLearnTransformer": _map_keras_wrappers_sk_learn_transformer,
    "regularizers.deserialize": _map_keras_regularizers_deserialize,
    "regularizers.get": _map_keras_regularizers_get,
    "regularizers.serialize": _map_keras_regularizers_serialize,
    "regularizers.L1": _map_keras_regularizers_l1,
    "regularizers.l1": _map_keras_regularizers_l1_ext,
    "regularizers.L1L2": _map_keras_regularizers_l1_l2,
    "regularizers.l1_l2": _map_keras_regularizers_l1_l2_ext,
    "regularizers.L2": _map_keras_regularizers_l2,
    "regularizers.l2": _map_keras_regularizers_l2_ext,
    "regularizers.OrthogonalRegularizer": _map_keras_regularizers__orthogonal_regularizer,
    "regularizers.orthogonal_regularizer": _map_keras_regularizers_orthogonal_regularizer,
    "regularizers.Regularizer": _map_keras_regularizers__regularizer,
    "callbacks.BackupAndRestore": _map_keras_callbacks__backup_and_restore,
    "callbacks.Callback": _map_keras_callbacks__callback,
    "callbacks.CallbackList": _map_keras_callbacks__callback_list,
    "callbacks.CSVLogger": _map_keras_callbacks_csv_logger,
    "callbacks.EarlyStopping": _map_keras_callbacks__early_stopping,
    "callbacks.History": _map_keras_callbacks__history,
    "callbacks.LambdaCallback": _map_keras_callbacks__lambda_callback,
    "callbacks.LearningRateScheduler": _map_keras_callbacks__learning_rate_scheduler,
    "callbacks.ModelCheckpoint": _map_keras_callbacks__model_checkpoint,
    "callbacks.ProgbarLogger": _map_keras_callbacks__progbar_logger,
    "callbacks.ReduceLROnPlateau": _map_keras_callbacks__reduce_lr_on_plateau,
    "callbacks.RemoteMonitor": _map_keras_callbacks__remote_monitor,
    "callbacks.SwapEMAWeights": _map_keras_callbacks__swap_ema_weights,
    "callbacks.TensorBoard": _map_keras_callbacks__tensor_board,
    "callbacks.TerminateOnNaN": _map_keras_callbacks__terminate_on_na_n,
    "optimizers.deserialize": _map_keras_optimizers_deserialize,
    "optimizers.get": _map_keras_optimizers_get,
    "optimizers.serialize": _map_keras_optimizers_serialize,
    "optimizers.Adadelta": _map_keras_optimizers__adadelta,
    "optimizers.Adafactor": _map_keras_optimizers__adafactor,
    "optimizers.Adagrad": _map_keras_optimizers__adagrad,
    "optimizers.Adam": _map_keras_optimizers__adam,
    "optimizers.Adamax": _map_keras_optimizers__adamax,
    "optimizers.AdamW": _map_keras_optimizers__adam_w,
    "optimizers.Ftrl": _map_keras_optimizers__ftrl,
    "optimizers.Lamb": _map_keras_optimizers__lamb,
    "optimizers.Lion": _map_keras_optimizers__lion,
    "optimizers.LossScaleOptimizer": _map_keras_optimizers__loss_scale_optimizer,
    "optimizers.Muon": _map_keras_optimizers__muon,
    "optimizers.Nadam": _map_keras_optimizers__nadam,
    "optimizers.Optimizer": _map_keras_optimizers__optimizer,
    "optimizers.RMSprop": _map_keras_optimizers_rm_sprop,
    "optimizers.SGD": _map_keras_optimizers_sgd,
    "optimizers.schedules.CosineDecay": _map_keras_optimizers_schedules__cosine_decay,
    "optimizers.schedules.CosineDecayRestarts": _map_keras_optimizers_schedules__cosine_decay_restarts,
    "optimizers.schedules.ExponentialDecay": _map_keras_optimizers_schedules__exponential_decay,
    "optimizers.schedules.InverseTimeDecay": _map_keras_optimizers_schedules__inverse_time_decay,
    "optimizers.schedules.LearningRateSchedule": _map_keras_optimizers_schedules__learning_rate_schedule,
    "optimizers.schedules.PiecewiseConstantDecay": _map_keras_optimizers_schedules__piecewise_constant_decay,
    "optimizers.schedules.PolynomialDecay": _map_keras_optimizers_schedules__polynomial_decay,
    "optimizers.schedules.deserialize": _map_keras_optimizers_schedules_deserialize,
    "optimizers.schedules.serialize": _map_keras_optimizers_schedules_serialize,
    "layers.TFSMLayer": _map_keras_layers_tfsm_layer,
    "layers.deserialize": _map_keras_layers_deserialize,
    "layers.serialize": _map_keras_layers_serialize,
    "layers.Activation": _map_keras_layers__activation,
    "layers.ELU": _map_keras_layers_elu,
    "layers.LeakyReLU": _map_keras_layers__leaky_re_lu,
    "layers.PReLU": _map_keras_layers_p_re_lu,
    "layers.ReLU": _map_keras_layers__re_lu,
    "layers.Softmax": _map_keras_layers__softmax,
    "layers.AdditiveAttention": _map_keras_layers__additive_attention,
    "layers.Attention": _map_keras_layers__attention,
    "layers.GroupQueryAttention": _map_keras_layers__group_query_attention,
    "layers.MultiHeadAttention": _map_keras_layers__multi_head_attention,
    "layers.Conv1D": _map_keras_layers__conv1_d,
    "layers.Convolution1D": _map_keras_layers__convolution1_d,
    "layers.Conv1DTranspose": _map_keras_layers__conv1_d_transpose,
    "layers.Convolution1DTranspose": _map_keras_layers__convolution1_d_transpose,
    "layers.Conv2D": _map_keras_layers__conv2_d,
    "layers.Convolution2D": _map_keras_layers__convolution2_d,
    "layers.Conv2DTranspose": _map_keras_layers__conv2_d_transpose,
    "layers.Convolution2DTranspose": _map_keras_layers__convolution2_d_transpose,
    "layers.Conv3D": _map_keras_layers__conv3_d,
    "layers.Convolution3D": _map_keras_layers__convolution3_d,
    "layers.Conv3DTranspose": _map_keras_layers__conv3_d_transpose,
    "layers.Convolution3DTranspose": _map_keras_layers__convolution3_d_transpose,
    "layers.DepthwiseConv1D": _map_keras_layers__depthwise_conv1_d,
    "layers.DepthwiseConv2D": _map_keras_layers__depthwise_conv2_d,
    "layers.SeparableConv1D": _map_keras_layers__separable_conv1_d,
    "layers.SeparableConvolution1D": _map_keras_layers__separable_convolution1_d,
    "layers.SeparableConv2D": _map_keras_layers__separable_conv2_d,
    "layers.SeparableConvolution2D": _map_keras_layers__separable_convolution2_d,
    "layers.Dense": _map_keras_layers__dense,
    "layers.EinsumDense": _map_keras_layers__einsum_dense,
    "layers.Embedding": _map_keras_layers__embedding,
    "layers.Identity": _map_keras_layers__identity,
    "layers.Input": _map_keras_layers__input,
    "layers.InputLayer": _map_keras_layers__input_layer,
    "layers.Lambda": _map_keras_layers__lambda,
    "layers.Masking": _map_keras_layers__masking,
    "layers.ReversibleEmbedding": _map_keras_layers__reversible_embedding,
    "layers.Wrapper": _map_keras_layers__wrapper,
    "layers.InputSpec": _map_keras_layers__input_spec,
    "layers.Layer": _map_keras_layers__layer,
    "layers.Add": _map_keras_layers__add,
    "layers.add": _map_keras_layers_add,
    "layers.Average": _map_keras_layers__average,
    "layers.average": _map_keras_layers_average,
    "layers.Concatenate": _map_keras_layers__concatenate,
    "layers.concatenate": _map_keras_layers_concatenate,
    "layers.Dot": _map_keras_layers__dot,
    "layers.dot": _map_keras_layers_dot,
    "layers.Maximum": _map_keras_layers__maximum,
    "layers.maximum": _map_keras_layers_maximum,
    "layers.Minimum": _map_keras_layers__minimum,
    "layers.minimum": _map_keras_layers_minimum,
    "layers.Multiply": _map_keras_layers__multiply,
    "layers.multiply": _map_keras_layers_multiply,
    "layers.Subtract": _map_keras_layers__subtract,
    "layers.subtract": _map_keras_layers_subtract,
    "layers.BatchNormalization": _map_keras_layers__batch_normalization,
    "layers.GroupNormalization": _map_keras_layers__group_normalization,
    "layers.LayerNormalization": _map_keras_layers__layer_normalization,
    "layers.RMSNormalization": _map_keras_layers_rms_normalization,
    "layers.SpectralNormalization": _map_keras_layers__spectral_normalization,
    "layers.UnitNormalization": _map_keras_layers__unit_normalization,
    "layers.AdaptiveAveragePooling1D": _map_keras_layers__adaptive_average_pooling1_d,
    "layers.AdaptiveAveragePooling2D": _map_keras_layers__adaptive_average_pooling2_d,
    "layers.AdaptiveAveragePooling3D": _map_keras_layers__adaptive_average_pooling3_d,
    "layers.AdaptiveMaxPooling1D": _map_keras_layers__adaptive_max_pooling1_d,
    "layers.AdaptiveMaxPooling2D": _map_keras_layers__adaptive_max_pooling2_d,
    "layers.AdaptiveMaxPooling3D": _map_keras_layers__adaptive_max_pooling3_d,
    "layers.AveragePooling1D": _map_keras_layers__average_pooling1_d,
    "layers.AvgPool1D": _map_keras_layers__avg_pool1_d,
    "layers.AveragePooling2D": _map_keras_layers__average_pooling2_d,
    "layers.AvgPool2D": _map_keras_layers__avg_pool2_d,
    "layers.AveragePooling3D": _map_keras_layers__average_pooling3_d,
    "layers.AvgPool3D": _map_keras_layers__avg_pool3_d,
    "layers.GlobalAveragePooling1D": _map_keras_layers__global_average_pooling1_d,
    "layers.GlobalAvgPool1D": _map_keras_layers__global_avg_pool1_d,
    "layers.GlobalAveragePooling2D": _map_keras_layers__global_average_pooling2_d,
    "layers.GlobalAvgPool2D": _map_keras_layers__global_avg_pool2_d,
    "layers.GlobalAveragePooling3D": _map_keras_layers__global_average_pooling3_d,
    "layers.GlobalAvgPool3D": _map_keras_layers__global_avg_pool3_d,
    "layers.GlobalMaxPool1D": _map_keras_layers__global_max_pool1_d,
    "layers.GlobalMaxPooling1D": _map_keras_layers__global_max_pooling1_d,
    "layers.GlobalMaxPool2D": _map_keras_layers__global_max_pool2_d,
    "layers.GlobalMaxPooling2D": _map_keras_layers__global_max_pooling2_d,
    "layers.GlobalMaxPool3D": _map_keras_layers__global_max_pool3_d,
    "layers.GlobalMaxPooling3D": _map_keras_layers__global_max_pooling3_d,
    "layers.MaxPool1D": _map_keras_layers__max_pool1_d,
    "layers.MaxPooling1D": _map_keras_layers__max_pooling1_d,
    "layers.MaxPool2D": _map_keras_layers__max_pool2_d,
    "layers.MaxPooling2D": _map_keras_layers__max_pooling2_d,
    "layers.MaxPool3D": _map_keras_layers__max_pool3_d,
    "layers.MaxPooling3D": _map_keras_layers__max_pooling3_d,
    "layers.CategoryEncoding": _map_keras_layers__category_encoding,
    "layers.Discretization": _map_keras_layers__discretization,
    "layers.HashedCrossing": _map_keras_layers__hashed_crossing,
    "layers.Hashing": _map_keras_layers__hashing,
    "layers.AugMix": _map_keras_layers__aug_mix,
    "layers.AutoContrast": _map_keras_layers__auto_contrast,
    "layers.CenterCrop": _map_keras_layers__center_crop,
    "layers.CutMix": _map_keras_layers__cut_mix,
    "layers.Equalization": _map_keras_layers__equalization,
    "layers.MaxNumBoundingBoxes": _map_keras_layers__max_num_bounding_boxes,
    "layers.MixUp": _map_keras_layers__mix_up,
    "layers.RandAugment": _map_keras_layers__rand_augment,
    "layers.RandomBrightness": _map_keras_layers__random_brightness,
    "layers.RandomColorDegeneration": _map_keras_layers__random_color_degeneration,
    "layers.RandomColorJitter": _map_keras_layers__random_color_jitter,
    "layers.RandomContrast": _map_keras_layers__random_contrast,
    "layers.RandomCrop": _map_keras_layers__random_crop,
    "layers.RandomElasticTransform": _map_keras_layers__random_elastic_transform,
    "layers.RandomErasing": _map_keras_layers__random_erasing,
    "layers.RandomFlip": _map_keras_layers__random_flip,
    "layers.RandomGaussianBlur": _map_keras_layers__random_gaussian_blur,
    "layers.RandomGrayscale": _map_keras_layers__random_grayscale,
    "layers.RandomHue": _map_keras_layers__random_hue,
    "layers.RandomInvert": _map_keras_layers__random_invert,
    "layers.RandomPerspective": _map_keras_layers__random_perspective,
    "layers.RandomPosterization": _map_keras_layers__random_posterization,
    "layers.RandomRotation": _map_keras_layers__random_rotation,
    "layers.RandomSaturation": _map_keras_layers__random_saturation,
    "layers.RandomSharpness": _map_keras_layers__random_sharpness,
    "layers.RandomShear": _map_keras_layers__random_shear,
    "layers.RandomTranslation": _map_keras_layers__random_translation,
    "layers.RandomZoom": _map_keras_layers__random_zoom,
    "layers.Resizing": _map_keras_layers__resizing,
    "layers.Solarization": _map_keras_layers__solarization,
    "layers.IntegerLookup": _map_keras_layers__integer_lookup,
    "layers.MelSpectrogram": _map_keras_layers__mel_spectrogram,
    "layers.Normalization": _map_keras_layers__normalization,
    "layers.Pipeline": _map_keras_layers__pipeline,
    "layers.Rescaling": _map_keras_layers__rescaling,
    "layers.STFTSpectrogram": _map_keras_layers_stft_spectrogram,
    "layers.StringLookup": _map_keras_layers__string_lookup,
    "layers.TextVectorization": _map_keras_layers__text_vectorization,
    "layers.ActivityRegularization": _map_keras_layers__activity_regularization,
    "layers.AlphaDropout": _map_keras_layers__alpha_dropout,
    "layers.Dropout": _map_keras_layers__dropout,
    "layers.GaussianDropout": _map_keras_layers__gaussian_dropout,
    "layers.GaussianNoise": _map_keras_layers__gaussian_noise,
    "layers.SpatialDropout1D": _map_keras_layers__spatial_dropout1_d,
    "layers.SpatialDropout2D": _map_keras_layers__spatial_dropout2_d,
    "layers.SpatialDropout3D": _map_keras_layers__spatial_dropout3_d,
    "layers.Cropping1D": _map_keras_layers__cropping1_d,
    "layers.Cropping2D": _map_keras_layers__cropping2_d,
    "layers.Cropping3D": _map_keras_layers__cropping3_d,
    "layers.Flatten": _map_keras_layers__flatten,
    "layers.Permute": _map_keras_layers__permute,
    "layers.RepeatVector": _map_keras_layers__repeat_vector,
    "layers.Reshape": _map_keras_layers__reshape,
    "layers.UpSampling1D": _map_keras_layers__up_sampling1_d,
    "layers.UpSampling2D": _map_keras_layers__up_sampling2_d,
    "layers.UpSampling3D": _map_keras_layers__up_sampling3_d,
    "layers.ZeroPadding1D": _map_keras_layers__zero_padding1_d,
    "layers.ZeroPadding2D": _map_keras_layers__zero_padding2_d,
    "layers.ZeroPadding3D": _map_keras_layers__zero_padding3_d,
    "layers.Bidirectional": _map_keras_layers__bidirectional,
    "layers.ConvLSTM1D": _map_keras_layers__conv_lstm1_d,
    "layers.ConvLSTM2D": _map_keras_layers__conv_lstm2_d,
    "layers.ConvLSTM3D": _map_keras_layers__conv_lstm3_d,
    "layers.GRU": _map_keras_layers_gru,
    "layers.GRUCell": _map_keras_layers_gru_cell,
    "layers.LSTM": _map_keras_layers_lstm,
    "layers.LSTMCell": _map_keras_layers_lstm_cell,
    "layers.RNN": _map_keras_layers_rnn,
    "layers.SimpleRNN": _map_keras_layers__simple_rnn,
    "layers.SimpleRNNCell": _map_keras_layers__simple_rnn_cell,
    "layers.StackedRNNCells": _map_keras_layers__stacked_rnn_cells,
    "layers.TimeDistributed": _map_keras_layers__time_distributed,
    "layers.FlaxLayer": _map_keras_layers__flax_layer,
    "layers.JaxLayer": _map_keras_layers__jax_layer,
    "layers.TorchModuleWrapper": _map_keras_layers__torch_module_wrapper,
    "tree.MAP_TO_NONE": _map_keras_tree_map_to_none,
    "tree.assert_same_paths": _map_keras_tree_assert_same_paths,
    "tree.assert_same_structure": _map_keras_tree_assert_same_structure,
    "tree.flatten": _map_keras_tree_flatten,
    "tree.flatten_with_path": _map_keras_tree_flatten_with_path,
    "tree.is_nested": _map_keras_tree_is_nested,
    "tree.lists_to_tuples": _map_keras_tree_lists_to_tuples,
    "tree.map_shape_structure": _map_keras_tree_map_shape_structure,
    "tree.map_structure": _map_keras_tree_map_structure,
    "tree.map_structure_up_to": _map_keras_tree_map_structure_up_to,
    "tree.pack_sequence_as": _map_keras_tree_pack_sequence_as,
    "tree.traverse": _map_keras_tree_traverse,
}
