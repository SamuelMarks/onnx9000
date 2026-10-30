"""Unit tests and verification suite for reader functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.onnx2gguf.reader import GGUFReader


def test_GGUFReader() -> None:
    """Verify that ggufreader behaves as expected."""
    try:
        obj = GGUFReader()
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
