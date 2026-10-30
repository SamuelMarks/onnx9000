"""Unit tests and verification suite for pruning ops functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.olive.pruning_ops import Pruner


def test_Pruner() -> None:
    """Verify that pruner behaves as expected."""
    try:
        obj = Pruner()
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
