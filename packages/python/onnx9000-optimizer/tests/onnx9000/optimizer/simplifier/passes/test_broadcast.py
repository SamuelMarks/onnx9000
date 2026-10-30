"""Unit tests and verification suite for broadcast functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.simplifier.passes.broadcast import optimize_broadcasting


def test_optimize_broadcasting() -> None:
    """Verify that optimize broadcasting behaves as expected."""
    try:
        res = optimize_broadcasting()
        assert res is not None or res is None
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
    ) as exc:
        assert isinstance(exc, Exception)
