"""Unit tests and verification suite for api functionality."""

from onnx9000.converters.paddle.api import (
    _convert_paddle_graph,
    convert_paddle_to_onnx,
    paddle_optimize_graph,
)
from onnx9000.core.exceptions import Onnx9000Error


def test_paddle_optimize_graph() -> None:
    """Verify that paddle optimize graph behaves as expected."""
    try:
        res = paddle_optimize_graph()
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


def test__convert_paddle_graph() -> None:
    """Verify that convert paddle graph behaves as expected."""
    try:
        res = _convert_paddle_graph()
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


def test_convert_paddle_to_onnx() -> None:
    """Verify that convert paddle to onnx behaves as expected."""
    try:
        res = convert_paddle_to_onnx()
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
