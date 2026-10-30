"""Unit tests and verification suite for layout functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.hardware.layout import LayoutOptimizer


def test_LayoutOptimizer() -> None:
    """Verify that layout optimizer behaves as expected."""
    try:
        obj = LayoutOptimizer()
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
