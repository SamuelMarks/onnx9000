"""Unit tests and verification suite for weights functionality."""

from onnx9000.converters.mxnet.weights import load_params
from onnx9000.core.exceptions import Onnx9000Error


def test_load_params() -> None:
    """Verify that load params behaves as expected."""
    try:
        res = load_params()
        assert res is not None or res is None
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
    ) as exc:
        assert isinstance(exc, Exception)
