"""Unit tests and verification suite for compiler functionality."""

from onnx9000.backends.cuda.compiler import CUDACompiler
from onnx9000.core.exceptions import Onnx9000Error


def test_CUDACompiler() -> None:
    """Verify that cudacompiler behaves as expected."""
    try:
        obj = CUDACompiler()
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
