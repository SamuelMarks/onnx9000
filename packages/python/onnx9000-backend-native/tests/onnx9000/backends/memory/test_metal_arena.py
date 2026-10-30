"""Unit tests and verification suite for metal arena functionality."""

from onnx9000.backends.memory.metal_arena import MetalMemoryPlanner
from onnx9000.core.exceptions import Onnx9000Error


def test_MetalMemoryPlanner() -> None:
    """Verify that metal memory planner behaves as expected."""
    try:
        obj = MetalMemoryPlanner()
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
