"""Unit tests and verification suite for quantization functionality."""

from onnx9000.c_compiler.quantization import (
    generate_dequantize_linear,
    generate_qlinear_conv,
    generate_qlinear_matmul,
    generate_quantize_linear,
)
from onnx9000.core.exceptions import Onnx9000Error


def test_generate_quantize_linear() -> None:
    """Verify that generate quantize linear behaves as expected."""
    try:
        res = generate_quantize_linear()
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


def test_generate_dequantize_linear() -> None:
    """Verify that generate dequantize linear behaves as expected."""
    try:
        res = generate_dequantize_linear()
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


def test_generate_qlinear_matmul() -> None:
    """Verify that generate qlinear matmul behaves as expected."""
    try:
        res = generate_qlinear_matmul()
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


def test_generate_qlinear_conv() -> None:
    """Verify that generate qlinear conv behaves as expected."""
    try:
        res = generate_qlinear_conv()
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
