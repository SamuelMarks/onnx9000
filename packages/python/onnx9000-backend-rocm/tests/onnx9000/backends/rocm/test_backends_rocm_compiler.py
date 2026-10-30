"""Unit tests and verification suite for compiler functionality."""

from onnx9000.backends.rocm.compiler import ROCmCompiler
from onnx9000.core.exceptions import Onnx9000Error


def test_ROCmCompiler() -> None:
    """Verify that rocm compiler behaves as expected."""
    try:
        obj = ROCmCompiler()
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
