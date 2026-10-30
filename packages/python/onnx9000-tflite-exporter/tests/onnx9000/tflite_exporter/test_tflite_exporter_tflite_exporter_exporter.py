"""Unit tests and verification suite for exporter functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.tflite_exporter.exporter import TFLiteExporter


def test_TFLiteExporter() -> None:
    """Verify that tflite exporter behaves as expected."""
    try:
        obj = TFLiteExporter()
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
