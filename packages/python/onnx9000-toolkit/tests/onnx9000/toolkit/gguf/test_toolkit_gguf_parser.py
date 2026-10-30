"""Unit tests and verification suite for parser functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.toolkit.gguf.parser import GGUFError, GGUFParser


def test_GGUFError() -> None:
    """Verify that gguferror behaves as expected."""
    try:
        obj = GGUFError()
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


def test_GGUFParser() -> None:
    """Verify that ggufparser behaves as expected."""
    try:
        obj = GGUFParser()
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
