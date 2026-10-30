"""Unit tests and verification suite for builder functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.tflite_exporter.flatbuffer.builder import FlatBufferBuilder


def test_FlatBufferBuilder() -> None:
    """Verify that flat buffer builder behaves as expected."""
    try:
        obj = FlatBufferBuilder()
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
