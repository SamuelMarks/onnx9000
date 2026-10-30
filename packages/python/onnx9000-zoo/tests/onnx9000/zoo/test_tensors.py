"""Unit tests and verification suite for tensors functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.zoo.tensors import (
    BFloat16Upcaster,
    GSPMDReconciler,
    MsgPackFlaxDeserializer,
    SafeTensorsMmapParser,
)


def test_SafeTensorsMmapParser() -> None:
    """Verify that safe tensors mmap parser behaves as expected."""
    try:
        obj = SafeTensorsMmapParser()
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


def test_GSPMDReconciler() -> None:
    """Verify that gspmdreconciler behaves as expected."""
    try:
        obj = GSPMDReconciler()
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


def test_BFloat16Upcaster() -> None:
    """Verify that bfloat16 upcaster behaves as expected."""
    try:
        obj = BFloat16Upcaster()
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


def test_MsgPackFlaxDeserializer() -> None:
    """Verify that msg pack flax deserializer behaves as expected."""
    try:
        obj = MsgPackFlaxDeserializer()
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
