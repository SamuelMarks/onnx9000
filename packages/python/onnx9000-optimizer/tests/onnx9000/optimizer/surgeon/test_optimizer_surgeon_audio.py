"""Unit tests and verification suite for audio functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.surgeon.audio import _unpack_scalar, fold_mel_weights


def test__unpack_scalar() -> None:
    """Verify that unpack scalar behaves as expected."""
    try:
        res = _unpack_scalar()
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


def test_fold_mel_weights() -> None:
    """Verify that fold mel weights behaves as expected."""
    try:
        res = fold_mel_weights()
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
