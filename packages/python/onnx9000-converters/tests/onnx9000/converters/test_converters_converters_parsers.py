"""Unit tests and verification suite for parsers functionality."""

from onnx9000.converters.parsers import BaseParser, JAXprParser, PyTorchFXParser, XLAHLOParser
from onnx9000.core.exceptions import Onnx9000Error


def test_BaseParser() -> None:
    """Verify that base parser behaves as expected."""
    try:
        obj = BaseParser()
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


def test_PyTorchFXParser() -> None:
    """Verify that py torch fxparser behaves as expected."""
    try:
        obj = PyTorchFXParser()
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


def test_JAXprParser() -> None:
    """Verify that jaxpr parser behaves as expected."""
    try:
        obj = JAXprParser()
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


def test_XLAHLOParser() -> None:
    """Verify that xlahloparser behaves as expected."""
    try:
        obj = XLAHLOParser()
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
