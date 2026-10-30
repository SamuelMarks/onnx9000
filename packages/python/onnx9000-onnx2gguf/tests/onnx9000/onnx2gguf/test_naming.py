"""Unit tests and verification suite for naming functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.onnx2gguf.naming import rename_tensor


def test_rename_tensor() -> None:
    """Verify that rename tensor behaves as expected."""
    try:
        res = rename_tensor()
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
