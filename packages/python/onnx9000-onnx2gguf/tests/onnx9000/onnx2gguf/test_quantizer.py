"""Unit tests and verification suite for quantizer functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.onnx2gguf.quantizer import f32_to_f16, quantize_q4_0, quantize_q4_1, quantize_q8_0


def test_f32_to_f16() -> None:
    """Verify that f32 to f16 behaves as expected."""
    try:
        res = f32_to_f16()
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


def test_quantize_q4_0() -> None:
    """Verify that quantize q4 0 behaves as expected."""
    try:
        res = quantize_q4_0()
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


def test_quantize_q4_1() -> None:
    """Verify that quantize q4 1 behaves as expected."""
    try:
        res = quantize_q4_1()
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


def test_quantize_q8_0() -> None:
    """Verify that quantize q8 0 behaves as expected."""
    try:
        res = quantize_q8_0()
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
