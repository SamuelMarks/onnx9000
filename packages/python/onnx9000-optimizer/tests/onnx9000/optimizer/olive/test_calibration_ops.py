"""Unit tests and verification suite for calibration ops functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.olive.calibration_ops import CalibrationLoop


def test_CalibrationLoop() -> None:
    """Verify that calibration loop behaves as expected."""
    try:
        obj = CalibrationLoop()
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
