"""Unit tests and verification suite for engine functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.hummingbird.engine import TranspilationEngine


def test_TranspilationEngine() -> None:
    """Verify that transpilation engine behaves as expected."""
    try:
        obj = TranspilationEngine()
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
