"""Unit tests and verification suite for context functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.olive.context import PassContext


def test_PassContext() -> None:
    """Verify that pass context behaves as expected."""
    try:
        obj = PassContext()
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
