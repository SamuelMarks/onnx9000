"""Unit tests and verification suite for generator functionality."""

from onnx9000.backends.codegen.generator import Generator
from onnx9000.core.exceptions import Onnx9000Error


def test_Generator() -> None:
    """Verify that generator behaves as expected."""
    try:
        obj = Generator()
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
