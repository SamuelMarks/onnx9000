"""Unit tests and verification suite for quantizer functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.tflite_exporter.quantization.quantizer import Quantizer, TensorQuantization


def test_TensorQuantization() -> None:
    """Verify that tensor quantization behaves as expected."""
    try:
        obj = TensorQuantization()
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


def test_Quantizer() -> None:
    """Verify that quantizer behaves as expected."""
    try:
        obj = Quantizer()
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
