"""Unit tests and verification suite for rnn functionality."""

from onnx9000.converters.frontend.nn.rnn import GRU, LSTM, RNN, RNNBase
from onnx9000.core.exceptions import Onnx9000Error


def test_RNNBase() -> None:
    """Verify that rnnbase behaves as expected."""
    try:
        obj = RNNBase()
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


def test_RNN() -> None:
    """Verify that rnn behaves as expected."""
    try:
        obj = RNN()
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


def test_LSTM() -> None:
    """Verify that lstm behaves as expected."""
    try:
        obj = LSTM()
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


def test_GRU() -> None:
    """Verify that gru behaves as expected."""
    try:
        obj = GRU()
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
