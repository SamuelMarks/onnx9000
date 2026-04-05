"""Tests for compiler more."""

import io
from unittest.mock import patch

from onnx9000.core.ir import Graph
from onnx9000.onnx2gguf.compiler import compile_gguf


@patch("onnx9000.onnx2gguf.compiler.extract_tokenizer_metadata")
def test_compile_gguf_overrides(mock_tok):
    """Provides functional implementation."""
    mock_tok.return_value = {
        "float_val": 3.14,
        "bool_val": True,
        "str_list": ["a", "b"],
        "float_list": [1.1, 2.2],
        "int_list": [1, 2],
    }
    g = Graph("test")
    buf = io.BytesIO()
    overrides = {"my_override": 4.2}
    compile_gguf(g, buf, overrides)
    assert buf.getvalue()
