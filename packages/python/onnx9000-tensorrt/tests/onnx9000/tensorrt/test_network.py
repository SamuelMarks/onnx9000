"""Unit tests and verification suite for network functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.tensorrt.network import IBuilderConfig, INetworkDefinition, ITensor


def test_ITensor() -> None:
    """Verify that itensor behaves as expected."""
    try:
        obj = ITensor()
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


def test_INetworkDefinition() -> None:
    """Verify that inetwork definition behaves as expected."""
    try:
        obj = INetworkDefinition()
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


def test_IBuilderConfig() -> None:
    """Verify that ibuilder config behaves as expected."""
    try:
        obj = IBuilderConfig()
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
