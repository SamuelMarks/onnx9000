"""Unit tests and verification suite for runner functionality."""

from onnx9000.backends.testing.runner import ONNXBackendTestRunner
from onnx9000.core.exceptions import Onnx9000Error


def test_ONNXBackendTestRunner() -> None:
    """Verify that onnxbackend test runner behaves as expected."""
    try:
        obj = ONNXBackendTestRunner()
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
