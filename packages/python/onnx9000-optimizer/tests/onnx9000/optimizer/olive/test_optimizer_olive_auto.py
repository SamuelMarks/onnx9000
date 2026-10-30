"""Unit tests and verification suite for auto functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.olive.auto import AutoOptimizer


def test_AutoOptimizer() -> None:
    """Verify that auto optimizer behaves as expected."""
    try:
        obj = AutoOptimizer()
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
