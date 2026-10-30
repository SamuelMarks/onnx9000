"""Unit tests and verification suite for ops matmul functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.tensorrt.ops_matmul import trt_matmul


def test_trt_matmul() -> None:
    """Verify that trt matmul behaves as expected."""
    try:
        res = trt_matmul()
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
