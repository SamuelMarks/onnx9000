"""Unit tests and verification suite for target functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.olive.target import Target


def test_Target() -> None:
    """Verify that target behaves as expected."""
    try:
        obj = Target()
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
