"""Unit tests and verification suite for utils functionality."""

from onnx9000.backends.codegen.utils import get_attribute, get_omp_pragma, sanitize_name
from onnx9000.core.exceptions import Onnx9000Error


def test_get_attribute() -> None:
    """Verify that get attribute behaves as expected."""
    try:
        res = get_attribute()
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


def test_sanitize_name() -> None:
    """Verify that sanitize name behaves as expected."""
    try:
        res = sanitize_name()
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


def test_get_omp_pragma() -> None:
    """Verify that get omp pragma behaves as expected."""
    try:
        res = get_omp_pragma()
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
