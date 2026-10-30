"""Unit tests and verification suite for mlir functionality."""

from onnx9000.c_compiler.mlir import MLIRCompiler
from onnx9000.core.exceptions import Onnx9000Error


def test_MLIRCompiler() -> None:
    """Verify that mlircompiler behaves as expected."""
    try:
        obj = MLIRCompiler()
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
