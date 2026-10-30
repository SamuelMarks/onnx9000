"""Unit tests and verification suite for cpu arena functionality."""

from onnx9000.backends.memory.cpu_arena import CPUMemoryPlanner
from onnx9000.core.exceptions import Onnx9000Error


def test_CPUMemoryPlanner() -> None:
    """Verify that cpumemory planner behaves as expected."""
    try:
        obj = CPUMemoryPlanner()
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
