"""Comprehensive unit tests for onnx9000.paddle2onnx and onnx9000_paddle2onnx."""

import onnx9000_paddle2onnx
import pytest
from onnx9000.core.ir import Graph
from onnx9000.paddle2onnx import PaddleToONNXGraphBuilder, convert, convert_paddle_to_onnx


def test_paddle2onnx_convert_and_builder() -> None:
    """Verify convert(), PaddleToONNXGraphBuilder, and convert_paddle_to_onnx()."""
    with pytest.raises(ValueError, match="Invalid model string"):
        convert("")
    assert convert("test") == "[ONNX-IR] from test"
    assert onnx9000_paddle2onnx.convert("resnet") == "[ONNX-IR] from resnet"

    builder = PaddleToONNXGraphBuilder("paddle_net")
    assert isinstance(builder.graph, Graph)

    g = convert_paddle_to_onnx(b"")
    assert isinstance(g, Graph)
