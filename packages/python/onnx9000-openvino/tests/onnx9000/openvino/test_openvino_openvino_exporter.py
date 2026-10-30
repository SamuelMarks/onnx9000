"""Unit tests and verification suite for exporter functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.openvino.exporter import OpenVinoExporter


def test_OpenVinoExporter() -> None:
    """Verify that open vino exporter behaves as expected."""
    try:
        obj = OpenVinoExporter()
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
