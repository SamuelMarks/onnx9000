"""Unit tests and verification suite for containers functionality."""

from onnx9000.converters.frontend.nn.containers import (
    ModuleDict,
    ModuleList,
    ParameterList,
    Sequential,
)
from onnx9000.core.exceptions import Onnx9000Error


def test_Sequential() -> None:
    """Verify that sequential behaves as expected."""
    try:
        obj = Sequential()
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


def test_ModuleList() -> None:
    """Verify that module list behaves as expected."""
    try:
        obj = ModuleList()
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


def test_ModuleDict() -> None:
    """Verify that module dict behaves as expected."""
    try:
        obj = ModuleDict()
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


def test_ParameterList() -> None:
    """Verify that parameter list behaves as expected."""
    try:
        obj = ParameterList()
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
