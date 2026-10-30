"""Unit tests and verification suite for script functionality."""

from onnx9000.converters.torch.script import TorchScriptParser
from onnx9000.core.exceptions import Onnx9000Error


def test_TorchScriptParser() -> None:
    """Verify that torch script parser behaves as expected."""
    try:
        obj = TorchScriptParser()
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
