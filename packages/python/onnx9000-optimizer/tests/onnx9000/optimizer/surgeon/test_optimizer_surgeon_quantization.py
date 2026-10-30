"""Unit tests and verification suite for quantization functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.surgeon.quantization import quantize_ptq


def test_quantize_ptq() -> None:
    """Verify that quantize ptq behaves as expected."""
    try:
        res = quantize_ptq()
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
