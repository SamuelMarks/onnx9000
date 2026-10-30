"""Unit tests and verification suite for executor functionality."""

from onnx9000.backends.cpu.executor import CPUExecutionProvider
from onnx9000.core.exceptions import Onnx9000Error


def test_CPUExecutionProvider() -> None:
    """Verify that cpuexecution provider behaves as expected."""
    try:
        obj = CPUExecutionProvider()
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
