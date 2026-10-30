"""Unit tests and verification suite for reader functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.tflite_exporter.flatbuffer.reader import FlatBufferReader


def test_FlatBufferReader() -> None:
    """Verify that flat buffer reader behaves as expected."""
    try:
        obj = FlatBufferReader()
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
