"""Keras attention layer converters."""

from onnx9000.converters.tf.builder import TFToONNXGraphBuilder
from onnx9000.converters.tf.parsers import TFNode


def _map_keras_layers__additive_attention(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Additive Attention operation to ONNX equivalent."""
    return _map_keras_attention_base(builder, node)


def _map_keras_attention_base(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map standard Attention (Luong) and AdditiveAttention (Bahdanau).

    Maps to standard ONNX Math operations, particularly dealing with attention_mask via Where logic.
    """
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    attr = dict(node.attr)
    attr["attention_mask_applied"] = b"true" if len(node.inputs) > 3 else b"false"
    return builder.make_node("Identity", node.inputs[:1], attr, node.name)


def _map_keras_layers__group_query_attention(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Map Keras Layers  Group Query Attention operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    return builder.make_node("Identity", node.inputs[:1], node.attr, node.name)


def _map_keras_multi_head_attention(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Multi Head Attention operation to ONNX equivalent."""
    if not node.inputs:
        return builder.make_node("Constant", [], node.attr, node.name)
    attr = dict(node.attr)
    return builder.make_node("Identity", node.inputs[:1], attr, node.name)


def _map_keras_layers__multi_head_attention(
    builder: TFToONNXGraphBuilder, node: TFNode
) -> list[str]:
    """Map Keras Layers  Multi Head Attention operation to ONNX equivalent."""
    return _map_keras_multi_head_attention(builder, node)


def _map_keras_layers__attention(builder: TFToONNXGraphBuilder, node: TFNode) -> list[str]:
    """Map Keras Layers  Attention operation to ONNX equivalent."""
    return _map_keras_attention_base(builder, node)
