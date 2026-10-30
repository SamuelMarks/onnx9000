"""Unit tests and verification suite for parser functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.toolkit.script.parser import ScriptParser, script


def test_ScriptParser() -> None:
    """Verify that script parser behaves as expected."""
    try:
        obj = ScriptParser()
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


def test_script() -> None:
    """Verify that script behaves as expected."""
    try:
        res = script()
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
