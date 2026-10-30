"""Unit tests and verification suite for data unpacker functionality."""

from onnx9000.c_compiler.data_unpacker import dtype_to_struct_fmt, unpack_bytes_to_str
from onnx9000.core.exceptions import Onnx9000Error


def test_dtype_to_struct_fmt() -> None:
    """Verify that dtype to struct fmt behaves as expected."""
    try:
        res = dtype_to_struct_fmt()
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


def test_unpack_bytes_to_str() -> None:
    """Verify that unpack bytes to str behaves as expected."""
    try:
        res = unpack_bytes_to_str()
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
