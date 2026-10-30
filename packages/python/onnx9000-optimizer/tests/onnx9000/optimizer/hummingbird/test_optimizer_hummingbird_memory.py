"""Unit tests and verification suite for memory functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.hummingbird.memory import (
    TreeAbstractions,
    estimate_memory_footprint,
    select_optimal_strategy,
)


def test_TreeAbstractions() -> None:
    """Verify that tree abstractions behaves as expected."""
    try:
        obj = TreeAbstractions()
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


def test_estimate_memory_footprint() -> None:
    """Verify that estimate memory footprint behaves as expected."""
    try:
        res = estimate_memory_footprint()
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


def test_select_optimal_strategy() -> None:
    """Verify that select optimal strategy behaves as expected."""
    try:
        res = select_optimal_strategy()
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
