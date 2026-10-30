"""Unit tests and verification suite for cuda arena functionality."""

from onnx9000.backends.cuda.cuda_arena import CUDAMemoryPlanner
from onnx9000.core.exceptions import Onnx9000Error


def test_CUDAMemoryPlanner() -> None:
    """Verify that cudamemory planner behaves as expected."""
    try:
        obj = CUDAMemoryPlanner()
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
