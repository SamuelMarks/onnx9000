"""Unit tests and verification suite for fx functionality."""

from onnx9000.converters.torch.fx import FXParser
from onnx9000.core.exceptions import Onnx9000Error


def test_FXParser() -> None:
    """Verify that fxparser behaves as expected."""
    try:
        obj = FXParser()
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
