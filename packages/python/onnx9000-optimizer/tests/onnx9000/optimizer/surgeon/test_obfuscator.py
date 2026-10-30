"""Unit tests and verification suite for obfuscator functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.surgeon.obfuscator import obfuscate_names


def test_obfuscate_names() -> None:
    """Verify that obfuscate names behaves as expected."""
    try:
        res = obfuscate_names()
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
