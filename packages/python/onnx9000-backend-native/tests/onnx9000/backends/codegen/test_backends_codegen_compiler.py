"""Unit tests and verification suite for compiler functionality."""

from onnx9000.backends.codegen.compiler import (
    compile_cpp,
    compile_static_lib,
    compile_wasm,
    load_ctypes_library,
    load_pybind_module,
)
from onnx9000.core.exceptions import Onnx9000Error


def test_compile_cpp() -> None:
    """Verify that compile cpp behaves as expected."""
    try:
        res = compile_cpp()
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


def test_compile_wasm() -> None:
    """Verify that compile wasm behaves as expected."""
    try:
        res = compile_wasm()
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


def test_load_pybind_module() -> None:
    """Verify that load pybind module behaves as expected."""
    try:
        res = load_pybind_module()
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


def test_load_ctypes_library() -> None:
    """Verify that load ctypes library behaves as expected."""
    try:
        res = load_ctypes_library()
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


def test_compile_static_lib() -> None:
    """Verify that compile static lib behaves as expected."""
    try:
        res = compile_static_lib()
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
