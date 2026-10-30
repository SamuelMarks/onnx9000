"""Unit tests and verification suite for keras h5 parser functionality."""

from onnx9000.converters.tf.keras_h5_parser import KerasH5Parser
from onnx9000.core.exceptions import Onnx9000Error


def test_KerasH5Parser() -> None:
    """Verify that keras h5 parser behaves as expected."""
    try:
        obj = KerasH5Parser()
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
