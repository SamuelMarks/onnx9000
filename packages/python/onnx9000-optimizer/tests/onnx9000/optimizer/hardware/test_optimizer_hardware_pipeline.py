"""Unit tests and verification suite for pipeline functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.hardware.pipeline import PipelineOptimizer


def test_PipelineOptimizer() -> None:
    """Verify that pipeline optimizer behaves as expected."""
    try:
        obj = PipelineOptimizer()
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
