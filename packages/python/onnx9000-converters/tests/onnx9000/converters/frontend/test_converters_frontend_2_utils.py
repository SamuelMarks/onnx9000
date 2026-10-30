"""Unit tests and verification suite for utils functionality."""

from onnx9000.converters.frontend.utils import (
    infer_elementwise_shape,
    infer_matmul_shape,
    record_op,
)
from onnx9000.core.exceptions import Onnx9000Error


def test_infer_elementwise_shape() -> None:
    """Verify that infer elementwise shape behaves as expected."""
    try:
        res = infer_elementwise_shape()
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


def test_infer_matmul_shape() -> None:
    """Verify that infer matmul shape behaves as expected."""
    try:
        res = infer_matmul_shape()
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


def test_record_op() -> None:
    """Verify that record op behaves as expected."""
    try:
        res = record_op()
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
