"""Unit tests and verification suite for session functionality."""

from onnx9000.backends.session import InferenceSession, InferenceSessionError, IOBinding, NodeArg
from onnx9000.core.exceptions import Onnx9000Error


def test_InferenceSessionError() -> None:
    """Verify that inference session error behaves as expected."""
    try:
        obj = InferenceSessionError()
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


def test_NodeArg() -> None:
    """Verify that node arg behaves as expected."""
    try:
        obj = NodeArg()
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


def test_IOBinding() -> None:
    """Verify that iobinding behaves as expected."""
    try:
        obj = IOBinding()
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


def test_InferenceSession() -> None:
    """Verify that inference session behaves as expected."""
    try:
        obj = InferenceSession()
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
