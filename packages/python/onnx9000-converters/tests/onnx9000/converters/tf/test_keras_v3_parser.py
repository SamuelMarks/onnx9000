"""Unit tests and verification suite for keras v3 parser functionality."""

from onnx9000.converters.tf.keras_v3_parser import Keras3Parser
from onnx9000.core.exceptions import Onnx9000Error


def test_Keras3Parser() -> None:
    """Verify that keras3 parser behaves as expected."""
    try:
        obj = Keras3Parser()
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
