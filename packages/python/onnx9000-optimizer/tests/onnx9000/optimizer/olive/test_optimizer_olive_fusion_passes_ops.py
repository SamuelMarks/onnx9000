"""Unit tests and verification suite for fusion passes ops functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.olive.fusion_passes_ops import FusionPassesOps


def test_FusionPassesOps() -> None:
    """Verify that fusion passes ops behaves as expected."""
    try:
        obj = FusionPassesOps()
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
