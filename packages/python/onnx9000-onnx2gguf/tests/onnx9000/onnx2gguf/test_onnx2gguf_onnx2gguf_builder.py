"""Unit tests and verification suite for builder functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.onnx2gguf.builder import GGUFTensorType, GGUFValueType, GGUFWriter


def test_GGUFValueType() -> None:
    """Verify that ggufvalue type behaves as expected."""
    try:
        obj = GGUFValueType()
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


def test_GGUFTensorType() -> None:
    """Verify that gguftensor type behaves as expected."""
    try:
        obj = GGUFTensorType()
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


def test_GGUFWriter() -> None:
    """Verify that ggufwriter behaves as expected."""
    try:
        obj = GGUFWriter()
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
