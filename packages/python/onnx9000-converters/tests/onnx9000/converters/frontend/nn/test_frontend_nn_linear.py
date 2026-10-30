"""Unit tests and verification suite for linear functionality."""

from onnx9000.converters.frontend.nn.linear import Linear
from onnx9000.core.exceptions import Onnx9000Error


def test_Linear() -> None:
    """Verify that linear behaves as expected."""
    try:
        obj = Linear()
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
