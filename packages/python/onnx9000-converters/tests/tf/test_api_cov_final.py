"""Tests the api cov final module functionality."""

from unittest.mock import patch

from onnx9000.converters.tf.api import convert_tf_to_onnx


def test_convert_tf_to_onnx_else_branch() -> None:
    """Tests the convert tf to onnx else branch functionality."""
    op = b"\x1a\x07unknown"
    node_data = b"\n" + bytes([len(op)]) + op
    graph_data = b"\x12" + bytes([len(node_data)]) + node_data
    model_data = b"\n" + bytes([len(graph_data)]) + graph_data
    from onnx9000.converters.tf.builder import TFToONNXGraphBuilder

    original_make_node = TFToONNXGraphBuilder.make_node

    def mock_make_node(self, op_type, inputs, attributes, name_prefix, num_outputs=1):
        """Test the mock make node functionality."""
        out = original_make_node(self, op_type, inputs, attributes, name_prefix, num_outputs)
        self.graph.tensors.clear()
        return out

    with patch("onnx9000.converters.tf.builder.TFToONNXGraphBuilder.make_node", new=mock_make_node):
        g = convert_tf_to_onnx(model_data)
        assert len(g.outputs) > 0
