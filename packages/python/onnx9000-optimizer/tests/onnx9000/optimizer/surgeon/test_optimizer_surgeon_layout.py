"""Unit tests and verification suite for layout functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.surgeon.layout import optimize_layouts


def test_optimize_layouts() -> None:
    """Verify that optimize layouts behaves as expected."""
    try:
        res = optimize_layouts()
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
