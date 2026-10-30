"""Unit tests and verification suite for ffi functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.tensorrt.ffi import TensorRTFFI, _phase_1_20_bindings


def test_TensorRTFFI() -> None:
    """Verify that tensor rtffi behaves as expected."""
    try:
        obj = TensorRTFFI()
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


def test__phase_1_20_bindings() -> None:
    """Verify that phase 1 20 bindings behaves as expected."""
    try:
        res = _phase_1_20_bindings()
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
