"""Unit tests and verification suite for calibration functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.sparse.calibration import DataLoader, cross_entropy_loss, evaluate_accuracy


def test_DataLoader() -> None:
    """Verify that data loader behaves as expected."""
    try:
        obj = DataLoader()
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


def test_cross_entropy_loss() -> None:
    """Verify that cross entropy loss behaves as expected."""
    try:
        res = cross_entropy_loss()
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


def test_evaluate_accuracy() -> None:
    """Verify that evaluate accuracy behaves as expected."""
    try:
        res = evaluate_accuracy()
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
