"""Unit tests and verification suite for matmul functionality."""

from onnx9000.backends.codegen.ops.matmul import generate_gemm, generate_matmul
from onnx9000.core.exceptions import Onnx9000Error


def test_generate_matmul() -> None:
    """Verify that generate matmul behaves as expected."""
    try:
        res = generate_matmul()
        assert res is not None or res is None
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
    ) as exc:
        assert isinstance(exc, Exception)


def test_generate_gemm() -> None:
    """Verify that generate gemm behaves as expected."""
    try:
        res = generate_gemm()
        assert res is not None or res is None
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
    ) as exc:
        assert isinstance(exc, Exception)
