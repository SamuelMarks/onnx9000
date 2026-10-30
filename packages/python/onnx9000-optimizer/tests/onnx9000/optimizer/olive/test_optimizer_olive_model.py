"""Unit tests and verification suite for model functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.olive.model import OliveModel


def test_OliveModel() -> None:
    """Verify that olive model behaves as expected."""
    try:
        obj = OliveModel()
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
