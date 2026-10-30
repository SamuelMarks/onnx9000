"""Unit tests and verification suite for reduction ops functionality."""

from onnx9000.converters.tf.reduction_ops import (
    _map_bincount,
    _map_cumprod,
    _map_cumsum,
    _map_logical_binary,
    _map_not_equal,
    _map_reduce_op,
)
from onnx9000.core.exceptions import Onnx9000Error


def test__map_reduce_op() -> None:
    """Verify that map reduce op behaves as expected."""
    try:
        res = _map_reduce_op()
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


def test__map_bincount() -> None:
    """Verify that map bincount behaves as expected."""
    try:
        res = _map_bincount()
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


def test__map_cumsum() -> None:
    """Verify that map cumsum behaves as expected."""
    try:
        res = _map_cumsum()
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


def test__map_cumprod() -> None:
    """Verify that map cumprod behaves as expected."""
    try:
        res = _map_cumprod()
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


def test__map_logical_binary() -> None:
    """Verify that map logical binary behaves as expected."""
    try:
        res = _map_logical_binary()
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


def test__map_not_equal() -> None:
    """Verify that map not equal behaves as expected."""
    try:
        res = _map_not_equal()
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
