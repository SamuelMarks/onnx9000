"""Module docstring."""

from onnx9000.core.ir import Graph
from onnx9000.onnx2gguf.compiler import compile_gguf
from unittest.mock import patch
import io


@patch("onnx9000.onnx2gguf.compiler.extract_tokenizer_metadata")
def test_compile_gguf_overrides(mock_tok):
    """Docstring."""
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
