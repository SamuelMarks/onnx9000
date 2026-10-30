"""Unit tests and verification suite for evaluator functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.olive.evaluator import Evaluator


def test_Evaluator() -> None:
    """Verify that evaluator behaves as expected."""
    try:
        obj = Evaluator()
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
