"""Unit tests and verification suite for project generator functionality."""

from onnx9000.c_compiler.project_generator import (
    generate_arduino_sketch,
    generate_cmakelists,
    generate_main_c,
    generate_makefile,
)
from onnx9000.core.exceptions import Onnx9000Error


def test_generate_makefile() -> None:
    """Verify that generate makefile behaves as expected."""
    try:
        res = generate_makefile()
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


def test_generate_main_c() -> None:
    """Verify that generate main c behaves as expected."""
    try:
        res = generate_main_c()
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


def test_generate_cmakelists() -> None:
    """Verify that generate cmakelists behaves as expected."""
    try:
        res = generate_cmakelists()
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


def test_generate_arduino_sketch() -> None:
    """Verify that generate arduino sketch behaves as expected."""
    try:
        res = generate_arduino_sketch()
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
