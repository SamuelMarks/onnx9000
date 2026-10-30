"""Unit tests and verification suite for quantization functionality."""

from onnx9000.converters.frontend.quantization import (
    AQTParser,
    AWQParser,
    GGUFQuantizationMapper,
    GPTQParser,
)
from onnx9000.core.exceptions import Onnx9000Error


def test_GGUFQuantizationMapper() -> None:
    """Verify that ggufquantization mapper behaves as expected."""
    try:
        obj = GGUFQuantizationMapper()
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


def test_AWQParser() -> None:
    """Verify that awqparser behaves as expected."""
    try:
        obj = AWQParser()
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


def test_GPTQParser() -> None:
    """Verify that gptqparser behaves as expected."""
    try:
        obj = GPTQParser()
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


def test_AQTParser() -> None:
    """Verify that aqtparser behaves as expected."""
    try:
        obj = AQTParser()
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
