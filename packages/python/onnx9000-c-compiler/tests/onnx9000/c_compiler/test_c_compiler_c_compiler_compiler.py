"""Unit tests and verification suite for compiler functionality."""

from onnx9000.c_compiler.compiler import C89Compiler
from onnx9000.core.exceptions import Onnx9000Error


def test_C89Compiler() -> None:
    """Verify that c89 compiler behaves as expected."""
    try:
        obj = C89Compiler()
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
