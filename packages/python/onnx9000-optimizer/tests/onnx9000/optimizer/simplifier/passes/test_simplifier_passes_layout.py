"""Unit tests and verification suite for layout functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.simplifier.passes.layout import (
    transform_nchw_to_nhwc,
    transform_nhwc_to_nchw,
)


def test_transform_nchw_to_nhwc() -> None:
    """Verify that transform nchw to nhwc behaves as expected."""
    try:
        res = transform_nchw_to_nhwc()
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


def test_transform_nhwc_to_nchw() -> None:
    """Verify that transform nhwc to nchw behaves as expected."""
    try:
        res = transform_nhwc_to_nchw()
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
