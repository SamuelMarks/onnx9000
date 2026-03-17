"""Test RNN."""

from onnx9000.converters.frontend.builder import GraphBuilder, Tracing
from onnx9000.converters.frontend.nn.rnn import GRU, LSTM, RNN
from onnx9000.converters.frontend.tensor import Tensor
from onnx9000.core.dtypes import DType


def test_rnn() -> None:
    """Tests the test_rnn functionality."""
    rnn = RNN(10, 20)
    lstm = LSTM(10, 20, bidirectional=True)
    gru = GRU(10, 20, bias=False)
    assert rnn.weight_ih_l0.shape == (20, 10)
    assert lstm.weight_ih_l0.shape == (40, 10)
    assert not hasattr(gru, "bias_ih_l0")
    builder = GraphBuilder("dummy")
    with Tracing(builder):
        x = Tensor((5, 3, 10), DType.FLOAT32, "x")
        (y1, h1) = rnn(x)
        (y2, h2) = lstm(x)
        (y3, h3) = gru(x)
    assert y1 is not None


def test_rnn_list_return() -> None:
    """Tests the test_rnn_list_return functionality."""
    from unittest.mock import patch

    from onnx9000.converters.frontend.nn.rnn import GRU, LSTM, RNN

    rnn = RNN(10, 20)
    lstm = LSTM(10, 20, bidirectional=True)
    gru = GRU(10, 20, bias=False)
    with patch("onnx9000.converters.frontend.utils.record_op", return_value=["a", "b", "c"]):
        (y1, h1) = rnn(None, hx="dummy_hx")
        (y2, h2) = lstm(None, hx=("d1", "d2"))
        (y3, h3) = gru(None, hx="dummy_hx")
        assert y1 == "a"
        assert h1 == "b"
        assert y2 == "a"
        assert h2 == ("b", "c")
        assert y3 == "a"
        assert h3 == "b"
