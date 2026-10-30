"""Exhaustive coverage test for Keras layer mappers."""

from onnx9000.converters.tf.builder import TFToONNXGraphBuilder
from onnx9000.converters.tf.keras_layers import KERAS_LAYERS_MAPPING, _map_keras_conv_base
from onnx9000.converters.tf.parsers import TFNode


def test_all_keras_layers_coverage() -> None:
    """Call every registered Keras layer mapper and assert valid output node emission."""
    builder = TFToONNXGraphBuilder("test")

    for name, mapper in KERAS_LAYERS_MAPPING.items():
        node = TFNode(
            name=f"test_{name}",
            op=name,
            inputs=["in1", "in2", "in3"],
            attr={"kernel_size": [3], "strides": [1], "padding": "valid"},
        )
        outputs = mapper(builder, node)
        assert len(outputs) >= 1


def test_keras_conv_base_coverage_gaps() -> None:
    """Test causal and custom padding branches in _map_keras_conv_base."""
    builder = TFToONNXGraphBuilder("test")

    node = TFNode(
        name="causal",
        op="Conv2D",
        inputs=["in"],
        attr={"padding": "causal", "kernel_size": [3], "dilation_rate": [1]},
    )
    out_causal = _map_keras_conv_base(builder, node)
    assert len(out_causal) == 1
    assert builder.graph.nodes[-1].op_type == "Conv"

    node2 = TFNode(name="unknown_pad", op="Conv2D", inputs=["in"], attr={"padding": "custom"})
    out_custom = _map_keras_conv_base(builder, node2)
    assert len(out_custom) == 1
    assert builder.graph.nodes[-1].attributes["auto_pad"] == b"CUSTOM"
