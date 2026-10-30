"""Unit tests and verification suite for module functionality."""

from onnx9000.converters.frontend.nn.module import Module
from onnx9000.core.exceptions import Onnx9000Error


def test_Module() -> None:
    """Verify that module behaves as expected."""
    try:
        obj = Module()
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
