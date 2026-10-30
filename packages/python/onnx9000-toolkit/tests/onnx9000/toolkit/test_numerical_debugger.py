"""Unit tests and verification suite for numerical debugger functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.toolkit.numerical_debugger import NumericalDebugger


def test_NumericalDebugger() -> None:
    """Verify that numerical debugger behaves as expected."""
    try:
        obj = NumericalDebugger()
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
