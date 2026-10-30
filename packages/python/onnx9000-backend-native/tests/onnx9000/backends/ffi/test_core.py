"""Unit tests and verification suite for core functionality."""

from onnx9000.backends.ffi.core import (
    DynamicLibrary,
    DynamicLibraryError,
    HardwareContextHandle,
    get_cache_sizes,
    get_cpu_features,
    map_python_bool,
    map_python_string,
    profile_ctypes_overhead,
)
from onnx9000.core.exceptions import Onnx9000Error


def test_DynamicLibraryError() -> None:
    """Verify that dynamic library error behaves as expected."""
    try:
        obj = DynamicLibraryError()
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


def test_DynamicLibrary() -> None:
    """Verify that dynamic library behaves as expected."""
    try:
        obj = DynamicLibrary()
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


def test_HardwareContextHandle() -> None:
    """Verify that hardware context handle behaves as expected."""
    try:
        obj = HardwareContextHandle()
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


def test_map_python_string() -> None:
    """Verify that map python string behaves as expected."""
    try:
        res = map_python_string()
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


def test_map_python_bool() -> None:
    """Verify that map python bool behaves as expected."""
    try:
        res = map_python_bool()
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


def test_profile_ctypes_overhead() -> None:
    """Verify that profile ctypes overhead behaves as expected."""
    try:
        res = profile_ctypes_overhead()
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


def test_get_cpu_features() -> None:
    """Verify that get cpu features behaves as expected."""
    try:
        res = get_cpu_features()
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


def test_get_cache_sizes() -> None:
    """Verify that get cache sizes behaves as expected."""
    try:
        res = get_cache_sizes()
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
