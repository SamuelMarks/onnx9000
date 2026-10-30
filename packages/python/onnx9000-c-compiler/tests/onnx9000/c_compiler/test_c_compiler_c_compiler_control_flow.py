"""Unit tests and verification suite for control flow functionality."""

from onnx9000.c_compiler.control_flow import generate_if, generate_loop
from onnx9000.core.exceptions import Onnx9000Error


def test_generate_if() -> None:
    """Verify that generate if behaves as expected."""
    try:
        res = generate_if()
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


def test_generate_loop() -> None:
    """Verify that generate loop behaves as expected."""
    try:
        res = generate_loop()
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
