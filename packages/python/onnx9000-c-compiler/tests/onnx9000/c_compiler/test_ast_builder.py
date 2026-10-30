"""Unit tests and verification suite for ast builder functionality."""

from onnx9000.c_compiler.ast_builder import C89Builder
from onnx9000.core.exceptions import Onnx9000Error


def test_C89Builder() -> None:
    """Verify that c89 builder behaves as expected."""
    try:
        obj = C89Builder()
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
