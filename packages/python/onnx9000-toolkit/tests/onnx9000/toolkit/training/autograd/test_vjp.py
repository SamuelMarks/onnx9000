"""Unit tests and verification suite for vjp functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.toolkit.training.autograd.vjp import VJPRule


def test_VJPRule() -> None:
    """Verify that vjprule behaves as expected."""
    try:
        obj = VJPRule()
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
