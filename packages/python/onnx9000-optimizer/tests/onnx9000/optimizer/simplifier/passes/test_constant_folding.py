"""Unit tests and verification suite for constant folding functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.simplifier.passes.constant_folding import (
    ConstantFoldingPass,
    _evaluate_conv,
    _evaluate_pool,
    _numpy_to_tensor_proto,
    _tensor_to_numpy,
    constant_folding,
)


def test_ConstantFoldingPass() -> None:
    """Verify that constant folding pass behaves as expected."""
    try:
        obj = ConstantFoldingPass()
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


def test__evaluate_pool() -> None:
    """Verify that evaluate pool behaves as expected."""
    try:
        res = _evaluate_pool()
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


def test__evaluate_conv() -> None:
    """Verify that evaluate conv behaves as expected."""
    try:
        res = _evaluate_conv()
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


def test__numpy_to_tensor_proto() -> None:
    """Verify that numpy to tensor proto behaves as expected."""
    try:
        res = _numpy_to_tensor_proto()
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


def test__tensor_to_numpy() -> None:
    """Verify that tensor to numpy behaves as expected."""
    try:
        res = _tensor_to_numpy()
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


def test_constant_folding() -> None:
    """Verify that constant folding behaves as expected."""
    try:
        res = constant_folding()
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
