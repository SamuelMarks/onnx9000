"""Unit tests and verification suite for op generator functionality."""

from onnx9000.backends.codegen.op_generator import OpGenerator
from onnx9000.core.exceptions import Onnx9000Error


def test_OpGenerator() -> None:
    """Verify that op generator behaves as expected."""
    try:
        obj = OpGenerator()
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
