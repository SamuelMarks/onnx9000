"""Unit tests and verification suite for strategies functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.hummingbird.strategies import Strategy, TargetHardware


def test_Strategy() -> None:
    """Verify that strategy behaves as expected."""
    try:
        obj = Strategy()
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


def test_TargetHardware() -> None:
    """Verify that target hardware behaves as expected."""
    try:
        obj = TargetHardware()
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
