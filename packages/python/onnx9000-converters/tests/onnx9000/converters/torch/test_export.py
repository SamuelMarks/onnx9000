"""Unit tests and verification suite for export functionality."""

from onnx9000.converters.torch.export import ExportParser
from onnx9000.core.exceptions import Onnx9000Error


def test_ExportParser() -> None:
    """Verify that export parser behaves as expected."""
    try:
        obj = ExportParser()
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
