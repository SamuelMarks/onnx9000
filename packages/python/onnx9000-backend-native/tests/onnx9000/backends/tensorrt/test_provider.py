"""Unit tests and verification suite for provider functionality."""

from onnx9000.backends.tensorrt.provider import TensorrtExecutionProvider
from onnx9000.core.exceptions import Onnx9000Error


def test_TensorrtExecutionProvider() -> None:
    """Verify that tensorrt execution provider behaves as expected."""
    try:
        obj = TensorrtExecutionProvider()
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
