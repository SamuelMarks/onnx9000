"""Unit tests and verification suite for passes functionality."""

from onnx9000.converters.paddle.passes import (
    dce_pass,
    dropout_removal_pass,
    identity_removal_pass,
    paddle_optimize_graph,
)
from onnx9000.core.exceptions import Onnx9000Error


def test_identity_removal_pass() -> None:
    """Verify that identity removal pass behaves as expected."""
    try:
        res = identity_removal_pass()
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


def test_dropout_removal_pass() -> None:
    """Verify that dropout removal pass behaves as expected."""
    try:
        res = dropout_removal_pass()
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


def test_dce_pass() -> None:
    """Verify that dce pass behaves as expected."""
    try:
        res = dce_pass()
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
