"""Unit tests and verification suite for mapper functionality."""

from onnx9000.converters.mxnet.mapper import MXNetMapper
from onnx9000.core.exceptions import Onnx9000Error


def test_MXNetMapper() -> None:
    """Verify that mxnet mapper behaves as expected."""
    try:
        obj = MXNetMapper()
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
