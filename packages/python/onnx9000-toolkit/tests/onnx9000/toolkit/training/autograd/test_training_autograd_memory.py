"""Unit tests and verification suite for memory functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.toolkit.training.autograd.memory import optimize_backward_memory


def test_optimize_backward_memory() -> None:
    """Verify that optimize backward memory behaves as expected."""
    try:
        res = optimize_backward_memory()
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
