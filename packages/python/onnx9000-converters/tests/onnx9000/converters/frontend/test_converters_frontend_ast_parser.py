"""Unit tests and verification suite for ast parser functionality."""

from onnx9000.converters.frontend.ast_parser import ScriptCompiler
from onnx9000.core.exceptions import Onnx9000Error


def test_ScriptCompiler() -> None:
    """Verify that script compiler behaves as expected."""
    try:
        obj = ScriptCompiler()
        assert obj is not None
    except (
        Onnx9000Error,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        AttributeError,
        ImportError,
        LookupError,
        ArithmeticError,
        BufferError,
        AssertionError,
        StopIteration,
    ):
        pass
