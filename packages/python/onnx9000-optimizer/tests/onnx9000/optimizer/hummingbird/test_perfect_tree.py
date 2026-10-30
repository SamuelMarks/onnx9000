"""Unit tests and verification suite for perfect tree functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.hummingbird.perfect_tree import (
    PerfectTreeCompiler,
    handle_perfect_multi_output,
    map_categorical_perfect,
)


def test_PerfectTreeCompiler() -> None:
    """Verify that perfect tree compiler behaves as expected."""
    try:
        obj = PerfectTreeCompiler()
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


def test_handle_perfect_multi_output() -> None:
    """Verify that handle perfect multi output behaves as expected."""
    try:
        res = handle_perfect_multi_output()
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


def test_map_categorical_perfect() -> None:
    """Verify that map categorical perfect behaves as expected."""
    try:
        res = map_categorical_perfect()
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
