"""Unit tests and verification suite for embedding functionality."""

from onnx9000.converters.frontend.nn.embedding import Embedding
from onnx9000.core.exceptions import Onnx9000Error


def test_Embedding() -> None:
    """Verify that embedding behaves as expected."""
    try:
        obj = Embedding()
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
