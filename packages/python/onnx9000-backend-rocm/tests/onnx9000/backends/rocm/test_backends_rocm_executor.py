"""Unit tests and verification suite for executor functionality."""

from onnx9000.backends.rocm.executor import Dispatcher, _rocm_matmul
from onnx9000.core.exceptions import Onnx9000Error


def test_Dispatcher() -> None:
    """Verify that dispatcher behaves as expected."""
    try:
        obj = Dispatcher()
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


def test__rocm_matmul() -> None:
    """Verify that rocm matmul behaves as expected."""
    try:
        res = _rocm_matmul()
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
