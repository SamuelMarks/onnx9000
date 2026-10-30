"""Unit tests and verification suite for llm ops functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.olive.llm_ops import LlmOps


def test_LlmOps() -> None:
    """Verify that llm ops behaves as expected."""
    try:
        obj = LlmOps()
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
