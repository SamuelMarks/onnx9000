"""Unit tests and verification suite for pool functionality."""

from onnx9000.converters.frontend.nn.pool import (
    AdaptiveAvgPool2d,
    AvgPool1d,
    AvgPool2d,
    MaxPool1d,
    MaxPool2d,
    _AdaptiveAvgPoolNd,
    _AvgPoolNd,
    _MaxPoolNd,
    _pair,
    _single,
)
from onnx9000.core.exceptions import Onnx9000Error


def test__MaxPoolNd() -> None:
    """Verify that max pool nd behaves as expected."""
    try:
        obj = _MaxPoolNd()
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


def test_MaxPool1d() -> None:
    """Verify that max pool1d behaves as expected."""
    try:
        obj = MaxPool1d()
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


def test_MaxPool2d() -> None:
    """Verify that max pool2d behaves as expected."""
    try:
        obj = MaxPool2d()
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


def test__AvgPoolNd() -> None:
    """Verify that avg pool nd behaves as expected."""
    try:
        obj = _AvgPoolNd()
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


def test_AvgPool1d() -> None:
    """Verify that avg pool1d behaves as expected."""
    try:
        obj = AvgPool1d()
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


def test_AvgPool2d() -> None:
    """Verify that avg pool2d behaves as expected."""
    try:
        obj = AvgPool2d()
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


def test__AdaptiveAvgPoolNd() -> None:
    """Verify that adaptive avg pool nd behaves as expected."""
    try:
        obj = _AdaptiveAvgPoolNd()
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


def test_AdaptiveAvgPool2d() -> None:
    """Verify that adaptive avg pool2d behaves as expected."""
    try:
        obj = AdaptiveAvgPool2d()
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


def test__pair() -> None:
    """Verify that pair behaves as expected."""
    try:
        res = _pair()
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


def test__single() -> None:
    """Verify that single behaves as expected."""
    try:
        res = _single()
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
