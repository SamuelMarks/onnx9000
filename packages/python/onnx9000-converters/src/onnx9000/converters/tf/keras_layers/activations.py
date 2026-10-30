"""Keras activation layer and function converters."""

from onnx9000.converters.tf.builder import TFToONNXGraphBuilder
from onnx9000.converters.tf.parsers import TFNode


def _map_keras_activation(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Activation operation to ONNX equivalent."""
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


def _map_keras_activations_deserialize(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Activations Deserialize operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_get(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Activations Get operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_serialize(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Activations Serialize operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_celu(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Activations Celu operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_elu(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Activations Elu operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_exponential(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Activations Exponential operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_gelu(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Activations Gelu operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_glu(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Activations Glu operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_hard_shrink(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Activations Hard Shrink operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_hard_sigmoid(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Activations Hard Sigmoid operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_hard_silu(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Activations Hard Silu operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_hard_swish(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Activations Hard Swish operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_hard_tanh(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Activations Hard Tanh operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_leaky_relu(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Activations Leaky Relu operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_linear(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Activations Linear operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_log_sigmoid(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Activations Log Sigmoid operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_log_softmax(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Activations Log Softmax operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_mish(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Activations Mish operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_relu(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Activations Relu operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_relu6(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Activations Relu6 operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_selu(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Activations Selu operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_sigmoid(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Activations Sigmoid operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_silu(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Activations Silu operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_swish(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Activations Swish operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_soft_shrink(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Activations Soft Shrink operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_softmax(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Activations Softmax operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_softplus(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Activations Softplus operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_softsign(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Activations Softsign operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_sparse_plus(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Activations Sparse Plus operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_sparse_sigmoid(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Activations Sparse Sigmoid operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_sparsemax(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Activations Sparsemax operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_squareplus(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Activations Squareplus operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_tanh(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Activations Tanh operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_tanh_shrink(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Activations Tanh Shrink operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_activations_threshold(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Activations Threshold operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__activation(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Activation operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers_elu(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers Elu operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__leaky_re_lu(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Leaky Re Lu operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers_p_re_lu(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers P Re Lu operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__re_lu(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Re Lu operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_layers__softmax(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Softmax operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)
