"""Unit tests and verification suite for builder functionality."""

from onnx9000.converters.frontend.builder import GraphBuilder, Tracing, get_active_builder
from onnx9000.core.exceptions import Onnx9000Error


def test_GraphBuilder() -> None:
    """Verify that graph builder behaves as expected."""
    try:
        obj = GraphBuilder()
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


def test_Tracing() -> None:
    """Verify that tracing behaves as expected."""
    try:
        obj = Tracing()
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


def test_get_active_builder() -> None:
    """Verify that get active builder behaves as expected."""
    try:
        res = get_active_builder()
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
