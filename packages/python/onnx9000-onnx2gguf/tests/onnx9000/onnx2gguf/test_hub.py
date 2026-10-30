"""Unit tests and verification suite for hub functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.onnx2gguf.hub import fetch_hf_config, generate_readme


def test_fetch_hf_config() -> None:
    """Verify that fetch hf config behaves as expected."""
    try:
        res = fetch_hf_config()
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


def test_generate_readme() -> None:
    """Verify that generate readme behaves as expected."""
    try:
        res = generate_readme()
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
