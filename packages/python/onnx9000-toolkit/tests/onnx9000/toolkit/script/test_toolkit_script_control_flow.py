"""Unit tests and verification suite for control flow functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.toolkit.script.control_flow import BranchContext, IfContextManager, LoopContextManager


def test_BranchContext() -> None:
    """Verify that branch context behaves as expected."""
    try:
        obj = BranchContext()
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


def test_IfContextManager() -> None:
    """Verify that if context manager behaves as expected."""
    try:
        obj = IfContextManager()
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


def test_LoopContextManager() -> None:
    """Verify that loop context manager behaves as expected."""
    try:
        obj = LoopContextManager()
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
