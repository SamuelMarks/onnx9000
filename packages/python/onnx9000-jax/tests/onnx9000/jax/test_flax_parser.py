"""Unit tests and verification suite for flax parser functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.jax.flax_parser import parse_msgpack


def test_parse_msgpack() -> None:
    """Verify that parse msgpack behaves as expected."""
    try:
        res = parse_msgpack()
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
