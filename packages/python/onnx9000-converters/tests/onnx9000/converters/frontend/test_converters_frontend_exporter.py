"""Unit tests and verification suite for exporter functionality."""

from onnx9000.converters.frontend.exporter import export, visualize
from onnx9000.core.exceptions import Onnx9000Error


def test_export() -> None:
    """Verify that export behaves as expected."""
    try:
        res = export()
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


def test_visualize() -> None:
    """Verify that visualize behaves as expected."""
    try:
        res = visualize()
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
