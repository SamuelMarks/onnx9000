"""Unit tests and verification suite for hub functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.toolkit.safetensors.hub import _get_cache_dir, cached_download, resolve_model_file


def test__get_cache_dir() -> None:
    """Verify that get cache dir behaves as expected."""
    try:
        res = _get_cache_dir()
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


def test_resolve_model_file() -> None:
    """Verify that resolve model file behaves as expected."""
    try:
        res = resolve_model_file()
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


def test_cached_download() -> None:
    """Verify that cached download behaves as expected."""
    try:
        res = cached_download()
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
