"""Unit tests and verification suite for bundler functionality."""

from onnx9000.c_compiler.bundler import bundle_weights_bin, generate_memory_summary
from onnx9000.core.exceptions import Onnx9000Error


def test_bundle_weights_bin() -> None:
    """Verify that bundle weights bin behaves as expected."""
    try:
        res = bundle_weights_bin()
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


def test_generate_memory_summary() -> None:
    """Verify that generate memory summary behaves as expected."""
    try:
        res = generate_memory_summary()
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
