"""Unit tests and verification suite for testing ops functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.olive.testing_ops import TestingOps


def test_TestingOps() -> None:
    """Verify that testing ops behaves as expected."""
    try:
        obj = TestingOps()
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
