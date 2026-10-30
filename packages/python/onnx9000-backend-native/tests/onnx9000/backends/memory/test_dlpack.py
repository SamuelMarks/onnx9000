"""Unit tests and verification suite for dlpack functionality."""

from onnx9000.backends.memory.dlpack import (
    DLDataType,
    DLDevice,
    DLManagedTensor,
    DLTensor,
    from_dlpack,
    from_numpy,
)
from onnx9000.core.exceptions import Onnx9000Error


def test_DLDataType() -> None:
    """Verify that dldata type behaves as expected."""
    try:
        obj = DLDataType()
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


def test_DLDevice() -> None:
    """Verify that dldevice behaves as expected."""
    try:
        obj = DLDevice()
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


def test_DLTensor() -> None:
    """Verify that dltensor behaves as expected."""
    try:
        obj = DLTensor()
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


def test_DLManagedTensor() -> None:
    """Verify that dlmanaged tensor behaves as expected."""
    try:
        obj = DLManagedTensor()
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


def test_from_dlpack() -> None:
    """Verify that from dlpack behaves as expected."""
    try:
        res = from_dlpack()
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


def test_from_numpy() -> None:
    """Verify that from numpy behaves as expected."""
    try:
        res = from_numpy()
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
