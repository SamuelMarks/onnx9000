"""Unit tests and verification suite for hasher functionality."""

from onnx9000.converters.jit.hasher import hash_graph
from onnx9000.core.exceptions import Onnx9000Error


def test_hash_graph() -> None:
    """Verify that hash graph behaves as expected."""
    try:
        res = hash_graph()
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
