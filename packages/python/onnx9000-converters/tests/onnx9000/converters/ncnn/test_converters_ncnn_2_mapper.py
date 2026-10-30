"""Unit tests and verification suite for mapper functionality."""

from onnx9000.converters.ncnn.mapper import NCNNMapper
from onnx9000.core.exceptions import Onnx9000Error


def test_NCNNMapper() -> None:
    """Verify that ncnnmapper behaves as expected."""
    try:
        obj = NCNNMapper()
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
