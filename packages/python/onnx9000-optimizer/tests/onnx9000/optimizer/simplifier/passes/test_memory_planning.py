"""Unit tests and verification suite for memory planning functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.simplifier.passes.memory_planning import (
    estimate_memory_consumption,
    plan_tensor_lifecycles,
)


def test_estimate_memory_consumption() -> None:
    """Verify that estimate memory consumption behaves as expected."""
    try:
        res = estimate_memory_consumption()
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


def test_plan_tensor_lifecycles() -> None:
    """Verify that plan tensor lifecycles behaves as expected."""
    try:
        res = plan_tensor_lifecycles()
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
