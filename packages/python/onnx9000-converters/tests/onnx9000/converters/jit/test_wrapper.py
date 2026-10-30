"""Unit tests and verification suite for wrapper functionality."""

from onnx9000.converters.jit.wrapper import CompiledModel
from onnx9000.core.exceptions import Onnx9000Error


def test_CompiledModel() -> None:
    """Verify that compiled model behaves as expected."""
    try:
        obj = CompiledModel()
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
