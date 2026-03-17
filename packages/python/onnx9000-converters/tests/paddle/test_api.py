"""Tests the api module functionality."""

from unittest.mock import patch

from onnx9000.converters.paddle.api import convert_paddle_to_onnx
from onnx9000.core.ir import Graph


def test_convert_paddle_to_onnx_empty() -> None:
    """Tests the convert paddle to onnx empty functionality."""
    g = convert_paddle_to_onnx(b"")
    assert isinstance(g, Graph)
    assert g.name == "paddle_graph"
    assert len(g.nodes) == 0


def test_convert_paddle_to_onnx_fallback(caplog) -> None:
    """Tests the convert paddle to onnx fallback functionality."""
    op = b"\x1a\x07unknown"
    block_data = b'\x08\x00\x10\x00"' + bytes([len(op)]) + op
    program_data = b"\n" + bytes([len(block_data)]) + block_data
    g = convert_paddle_to_onnx(program_data)
    assert any(n.op_type == "Custom_Paddle_unknown" for n in g.nodes)
    assert "Fallback to custom op for unknown Paddle node: unknown" in caplog.text


def test_convert_paddle_to_onnx_basic_graph() -> None:
    """Tests the convert paddle to onnx basic graph functionality."""
    op1 = b"\x1a\x04feed\n\x0b\n\x03Out\x12\x04in_1"
    op2 = b"\x1a\x04relu\n\x0c\n\x01X\x12\x04in_1\x12\x0b\n\x03Out\x12\x04out1"
    block_data = b'\x08\x00\x10\x00"' + bytes([len(op1)]) + op1 + b'"' + bytes([len(op2)]) + op2
    program_data = b"\n" + bytes([len(block_data)]) + block_data
    g = convert_paddle_to_onnx(program_data, params_data=b"mock")
    op_types = [n.op_type for n in g.nodes]
    assert "Relu" in op_types
    assert "Custom_Paddle_feed" in op_types


def test_convert_paddle_to_onnx_else_branch() -> None:
    """Tests the convert paddle to onnx else branch functionality."""
    op = b"\x1a\x07unknown"
    block_data = b'\x08\x00\x10\x00"' + bytes([len(op)]) + op
    program_data = b"\n" + bytes([len(block_data)]) + block_data
    from onnx9000.converters.paddle.builder import PaddleToONNXGraphBuilder

    original_make_node = PaddleToONNXGraphBuilder.make_node

    def mock_make_node(self, op_type, inputs, attributes, name_prefix, outputs=None):
        """Tests the mock make node functionality."""
        out = original_make_node(self, op_type, inputs, attributes, name_prefix, outputs)
        self.graph.tensors.clear()
        return out

    with patch(
        "onnx9000.converters.paddle.builder.PaddleToONNXGraphBuilder.make_node", new=mock_make_node
    ):
        g = convert_paddle_to_onnx(program_data)
        assert len(g.outputs) > 0
