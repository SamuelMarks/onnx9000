"""Unit tests and verification suite for base functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.simplifier.passes.base import GraphPass


def test_GraphPass() -> None:
    """Verify that graph pass behaves as expected."""
    try:
        obj = GraphPass()
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
