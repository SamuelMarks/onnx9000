import pytest

"""Tests the api module functionality."""

from onnx9000.converters.tf.api import (
    convert_keras_to_onnx,
    convert_tf_to_onnx,
    convert_tflite_to_onnx,
)
from onnx9000.core.ir import Graph


def test_convert_tf_to_onnx_graphdef() -> None:
    """Tests the convert tf to onnx graphdef functionality."""
    node1 = b"\n\x02in\x12\x0bPlaceholder"
    node2 = b"\n\x04relu\x12\x04Relu\x1a\x02in"
    data = b"\n" + bytes([len(node1)]) + node1 + b"\n" + bytes([len(node2)]) + node2
    graph = convert_tf_to_onnx(data)
    assert isinstance(graph, Graph)
    assert graph.name == "tf_graph"
    assert any(n.op_type == "Relu" for n in graph.nodes)


def test_convert_tf_to_onnx_saved_model() -> None:
    """Tests the convert tf to onnx saved model functionality."""
    graph = convert_tf_to_onnx(b"", is_saved_model=True)
    assert isinstance(graph, Graph)


@pytest.mark.skip("keras not installed")
def test_convert_keras_to_onnx() -> None:
    """Tests the convert keras to onnx functionality."""
    g1 = convert_keras_to_onnx(b"")
    assert isinstance(g1, Graph)
    assert g1.name == "keras_graph"
    g2 = convert_keras_to_onnx(b"", is_v3=True)
    assert isinstance(g2, Graph)
    assert g2.name == "keras_graph"


def test_convert_tflite_to_onnx() -> None:
    """Tests the convert tflite to onnx functionality."""
    g = convert_tflite_to_onnx(b"")
    assert isinstance(g, Graph)
    assert g.name == "tflite_graph"


def test_fallback_op_conversion(caplog) -> None:
    """Tests the fallback op conversion functionality."""
    node1 = b"\n\x04test\x12\x0bUnknownOpXX"
    data = b"\n" + bytes([len(node1)]) + node1
    graph = convert_tf_to_onnx(data)
    assert any(n.op_type == "Custom_TF_UnknownOpXX" for n in graph.nodes)
    assert "Fallback to custom op for unknown node: UnknownOpXX" in caplog.text
