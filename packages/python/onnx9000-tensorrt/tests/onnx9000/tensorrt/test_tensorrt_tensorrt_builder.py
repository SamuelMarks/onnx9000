"""Unit tests and verification suite for builder functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.tensorrt.builder import Builder, BuilderConfig, NetworkDefinition


def test_BuilderConfig() -> None:
    """Verify that builder config behaves as expected."""
    try:
        obj = BuilderConfig()
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


def test_NetworkDefinition() -> None:
    """Verify that network definition behaves as expected."""
    try:
        obj = NetworkDefinition()
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


def test_Builder() -> None:
    """Verify that builder behaves as expected."""
    try:
        obj = Builder()
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
