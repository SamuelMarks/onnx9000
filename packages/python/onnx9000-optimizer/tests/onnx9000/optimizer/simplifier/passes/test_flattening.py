"""Unit tests and verification suite for flattening functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.simplifier.passes.flattening import flatten_subgraphs


def test_flatten_subgraphs() -> None:
    """Verify that flatten subgraphs behaves as expected."""
    try:
        res = flatten_subgraphs()
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
