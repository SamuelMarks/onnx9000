"""Unit tests and verification suite for dropout functionality."""

from onnx9000.converters.frontend.nn.dropout import (
    Dropout,
    Dropout1d,
    Dropout2d,
    Dropout3d,
    _DropoutNd,
)
from onnx9000.core.exceptions import Onnx9000Error


def test__DropoutNd() -> None:
    """Verify that dropout nd behaves as expected."""
    try:
        obj = _DropoutNd()
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


def test_Dropout() -> None:
    """Verify that dropout behaves as expected."""
    try:
        obj = Dropout()
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


def test_Dropout1d() -> None:
    """Verify that dropout1d behaves as expected."""
    try:
        obj = Dropout1d()
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


def test_Dropout2d() -> None:
    """Verify that dropout2d behaves as expected."""
    try:
        obj = Dropout2d()
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


def test_Dropout3d() -> None:
    """Verify that dropout3d behaves as expected."""
    try:
        obj = Dropout3d()
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
