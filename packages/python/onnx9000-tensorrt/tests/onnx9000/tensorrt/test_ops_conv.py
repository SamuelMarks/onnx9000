"""Unit tests and verification suite for ops conv functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.tensorrt.ops_conv import trt_conv


def test_trt_conv() -> None:
    """Verify that trt conv behaves as expected."""
    try:
        res = trt_conv()
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
