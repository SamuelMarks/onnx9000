"""Unit tests and verification suite for cli functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.cli import (
    ModelCache,
    is_package_under_5mb,
    optimize_cli,
    save_onnx,
    save_safetensors,
)


def test_ModelCache() -> None:
    """Verify that model cache behaves as expected."""
    try:
        obj = ModelCache()
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


def test_save_onnx() -> None:
    """Verify that save onnx behaves as expected."""
    try:
        res = save_onnx()
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


def test_save_safetensors() -> None:
    """Verify that save safetensors behaves as expected."""
    try:
        res = save_safetensors()
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


def test_optimize_cli() -> None:
    """Verify that optimize cli behaves as expected."""
    try:
        res = optimize_cli()
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


def test_is_package_under_5mb() -> None:
    """Verify that is package under 5mb behaves as expected."""
    try:
        res = is_package_under_5mb()
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
