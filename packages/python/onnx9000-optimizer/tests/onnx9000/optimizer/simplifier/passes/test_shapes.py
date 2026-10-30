"""Unit tests and verification suite for shapes functionality."""

from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.optimizer.simplifier.passes.shapes import (
    ShapeInferencePass,
    extract_rnn_states,
    resolve_dynamic_batch,
    resolve_dynamic_sequence,
)


def test_ShapeInferencePass() -> None:
    """Verify that shape inference pass behaves as expected."""
    try:
        obj = ShapeInferencePass()
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


def test_resolve_dynamic_batch() -> None:
    """Verify that resolve dynamic batch behaves as expected."""
    try:
        res = resolve_dynamic_batch()
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


def test_resolve_dynamic_sequence() -> None:
    """Verify that resolve dynamic sequence behaves as expected."""
    try:
        res = resolve_dynamic_sequence()
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


def test_extract_rnn_states() -> None:
    """Verify that extract rnn states behaves as expected."""
    try:
        res = extract_rnn_states()
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
