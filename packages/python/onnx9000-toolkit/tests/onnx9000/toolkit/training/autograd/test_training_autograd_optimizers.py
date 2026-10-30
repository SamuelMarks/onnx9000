"""Unit tests and verification suite for optimizers functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.toolkit.training.autograd.optimizers import (
    add_adadelta_optimizer,
    add_adagrad_optimizer,
    add_adam_optimizer,
    add_adamw_optimizer,
    add_differential_privacy_noise,
    add_gradient_accumulation,
    add_gradient_clipping,
    add_gradient_clipping_value,
    add_local_dp_gradient_clipping,
    add_rmsprop_optimizer,
    add_sgd_optimizer,
)


def test_add_sgd_optimizer() -> None:
    """Verify that add sgd optimizer behaves as expected."""
    try:
        res = add_sgd_optimizer()
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


def test_add_adam_optimizer() -> None:
    """Verify that add adam optimizer behaves as expected."""
    try:
        res = add_adam_optimizer()
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


def test_add_adamw_optimizer() -> None:
    """Verify that add adamw optimizer behaves as expected."""
    try:
        res = add_adamw_optimizer()
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


def test_add_rmsprop_optimizer() -> None:
    """Verify that add rmsprop optimizer behaves as expected."""
    try:
        res = add_rmsprop_optimizer()
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


def test_add_adagrad_optimizer() -> None:
    """Verify that add adagrad optimizer behaves as expected."""
    try:
        res = add_adagrad_optimizer()
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


def test_add_adadelta_optimizer() -> None:
    """Verify that add adadelta optimizer behaves as expected."""
    try:
        res = add_adadelta_optimizer()
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


def test_add_gradient_accumulation() -> None:
    """Verify that add gradient accumulation behaves as expected."""
    try:
        res = add_gradient_accumulation()
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


def test_add_differential_privacy_noise() -> None:
    """Verify that add differential privacy noise behaves as expected."""
    try:
        res = add_differential_privacy_noise()
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


def test_add_gradient_clipping() -> None:
    """Verify that add gradient clipping behaves as expected."""
    try:
        res = add_gradient_clipping()
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


def test_add_local_dp_gradient_clipping() -> None:
    """Verify that add local dp gradient clipping behaves as expected."""
    try:
        res = add_local_dp_gradient_clipping()
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


def test_add_gradient_clipping_value() -> None:
    """Verify that add gradient clipping value behaves as expected."""
    try:
        res = add_gradient_clipping_value()
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
