"""Keras normalization and regularization layer converters."""

from collections.abc import Callable

from onnx9000.converters.tf.builder import TFToONNXGraphBuilder
from onnx9000.converters.tf.parsers import TFNode


def _map_keras_batch_normalization(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Batch Normalization operation to ONNX equivalent."""
    return builder.make_node("BatchNormalization", node.inputs, node.attr, node.name)


def _map_keras_layer_normalization(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layer Normalization operation to ONNX equivalent."""
    return builder.make_node("LayerNormalization", node.inputs, node.attr, node.name)


def _map_keras_dropout(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Dropout operation to ONNX equivalent."""
    return builder.make_node("Dropout", node.inputs, node.attr, node.name)


def _map_keras_spatial_dropout(dim: str) -> Callable:
    """Map Keras Spatial Dropout operation to ONNX equivalent."""

    def _impl(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
        """Execute the  impl operation."""
        return builder.make_node("Dropout", node.inputs, node.attr, node.name)

    return _impl


def _map_keras_constraints_deserialize(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Constraints Deserialize operation to ONNX equivalent."""
    return _map_keras_constraints(builder, node)


def _map_keras_constraints_get(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Constraints Get operation to ONNX equivalent."""
    return _map_keras_constraints(builder, node)


def _map_keras_constraints_serialize(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Constraints Serialize operation to ONNX equivalent."""
    return _map_keras_constraints(builder, node)


def _map_keras_constraints(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Constraints operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    attr = dict(node.attr)
    attr["keras_constraint"] = getattr(node, "op", "Constraint").encode("utf-8")
    return builder.make_node("Identity", node.inputs[:1], attr, node.name)


def _map_keras_constraints__max_norm(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Constraints  Max Norm operation to ONNX equivalent."""
    return _map_keras_constraints(builder, node)


def _map_keras_constraints_max_norm(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Constraints Max Norm operation to ONNX equivalent."""
    return _map_keras_constraints(builder, node)


def _map_keras_constraints__min_max_norm(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Constraints  Min Max Norm operation to ONNX equivalent."""
    return _map_keras_constraints(builder, node)


def _map_keras_constraints_min_max_norm(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Constraints Min Max Norm operation to ONNX equivalent."""
    return _map_keras_constraints(builder, node)


def _map_keras_constraints__non_neg(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Constraints  Non Neg operation to ONNX equivalent."""
    return _map_keras_constraints(builder, node)


def _map_keras_constraints_non_neg(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Constraints Non Neg operation to ONNX equivalent."""
    return _map_keras_constraints(builder, node)


def _map_keras_constraints__unit_norm(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Constraints  Unit Norm operation to ONNX equivalent."""
    return _map_keras_constraints(builder, node)


def _map_keras_constraints_unit_norm(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Constraints Unit Norm operation to ONNX equivalent."""
    return _map_keras_constraints(builder, node)


def _map_keras_layers__batch_normalization(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Map Keras Layers  Batch Normalization operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__group_normalization(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Map Keras Layers  Group Normalization operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__layer_normalization(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Map Keras Layers  Layer Normalization operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers_rms_normalization(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers Rms Normalization operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__spectral_normalization(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Map Keras Layers  Spectral Normalization operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__unit_normalization(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Unit Normalization operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__activity_regularization(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Map Keras Layers  Activity Regularization operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__alpha_dropout(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Alpha Dropout operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__dropout(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Dropout operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__gaussian_dropout(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Gaussian Dropout operation to ONNX equivalent."""
    return _map_keras_gaussian_dropout(builder, node)


def _map_keras_gaussian_noise(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Gaussian Noise operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_gaussian_dropout(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Gaussian Dropout operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__spatial_dropout1_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Spatial Dropout1 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__spatial_dropout2_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Spatial Dropout2 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__spatial_dropout3_d(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Spatial Dropout3 D operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__gaussian_noise(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Gaussian Noise operation to ONNX equivalent."""
    return _map_keras_gaussian_noise(builder, node)


def _map_keras_constraints__constraint(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Constraints  Constraint operation to ONNX equivalent."""
    return _map_keras_constraints(builder, node)
