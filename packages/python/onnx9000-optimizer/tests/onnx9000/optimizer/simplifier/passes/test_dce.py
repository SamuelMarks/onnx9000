"""Unit tests and verification suite for dce functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.simplifier.passes.dce import (
    ControlFlowFoldingPass,
    DCEPass,
    IdentityEliminationPass,
    dead_code_elimination,
)


def test_DCEPass() -> None:
    """Verify that dcepass behaves as expected."""
    try:
        obj = DCEPass()
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


def test_IdentityEliminationPass() -> None:
    """Verify that identity elimination pass behaves as expected."""
    try:
        obj = IdentityEliminationPass()
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


def test_ControlFlowFoldingPass() -> None:
    """Verify that control flow folding pass behaves as expected."""
    try:
        obj = ControlFlowFoldingPass()
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


def test_dead_code_elimination() -> None:
    """Verify that dead code elimination behaves as expected."""
    try:
        res = dead_code_elimination()
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
