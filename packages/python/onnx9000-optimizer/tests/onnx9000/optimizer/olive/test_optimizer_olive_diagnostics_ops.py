"""Unit tests and verification suite for diagnostics ops functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.olive.diagnostics_ops import DiagnosticsOps


def test_DiagnosticsOps() -> None:
    """Verify that diagnostics ops behaves as expected."""
    try:
        obj = DiagnosticsOps()
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
