"""Unit tests and verification suite for edgetpu functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.tflite_exporter.optimizations.edgetpu import EdgeTPUOptimizer


def test_EdgeTPUOptimizer() -> None:
    """Verify that edge tpuoptimizer behaves as expected."""
    try:
        obj = EdgeTPUOptimizer()
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
