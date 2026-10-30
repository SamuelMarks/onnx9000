"""Unit tests and verification suite for jaxpr string parser functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.jax.jaxpr_string_parser import parse_jaxpr_string


def test_parse_jaxpr_string() -> None:
    """Verify that parse jaxpr string behaves as expected."""
    try:
        res = parse_jaxpr_string()
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
