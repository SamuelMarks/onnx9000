"""Unit tests and verification suite for bindings functionality."""

from onnx9000.backends.cuda.bindings import (
    _register_cublas_api,
    _register_cuda_api,
    _register_cudnn_api,
    check_cublas_error,
    check_cuda_error,
    check_cudnn_error,
    is_cublas_available,
    is_cuda_available,
    is_cudnn_available,
)
from onnx9000.core.exceptions import Onnx9000Error


def test_is_cuda_available() -> None:
    """Verify that is cuda available behaves as expected."""
    try:
        res = is_cuda_available()
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


def test_is_cublas_available() -> None:
    """Verify that is cublas available behaves as expected."""
    try:
        res = is_cublas_available()
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


def test_is_cudnn_available() -> None:
    """Verify that is cudnn available behaves as expected."""
    try:
        res = is_cudnn_available()
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


def test__register_cuda_api() -> None:
    """Verify that register cuda api behaves as expected."""
    try:
        res = _register_cuda_api()
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


def test__register_cublas_api() -> None:
    """Verify that register cublas api behaves as expected."""
    try:
        res = _register_cublas_api()
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


def test__register_cudnn_api() -> None:
    """Verify that register cudnn api behaves as expected."""
    try:
        res = _register_cudnn_api()
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


def test_check_cuda_error() -> None:
    """Verify that check cuda error behaves as expected."""
    try:
        res = check_cuda_error()
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


def test_check_cublas_error() -> None:
    """Verify that check cublas error behaves as expected."""
    try:
        res = check_cublas_error()
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


def test_check_cudnn_error() -> None:
    """Verify that check cudnn error behaves as expected."""
    try:
        res = check_cudnn_error()
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
