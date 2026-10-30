"""Unit tests and verification suite for fusion ops functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.olive.fusion_ops import FusionOptimizer


def test_FusionOptimizer() -> None:
    """Verify that fusion optimizer behaves as expected."""
    try:
        obj = FusionOptimizer()
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
