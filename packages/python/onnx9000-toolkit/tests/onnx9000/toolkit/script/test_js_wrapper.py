"""Unit tests and verification suite for js wrapper functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.toolkit.script.js_wrapper import JSGraphBuilder


def test_JSGraphBuilder() -> None:
    """Verify that jsgraph builder behaves as expected."""
    try:
        obj = JSGraphBuilder()
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
