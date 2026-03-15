"""Test RNN."""

from onnx9000 import GraphBuilder, Tracing
from onnx9000.frontends.frontend.nn.rnn import RNN, LSTM, GRU
from onnx9000.frontends.frontend.tensor import Tensor
from onnx9000.core.dtypes import DType


def test_rnn():
    """Provides semantic functionality and verification."""
    rnn = RNN(10, 20)
    lstm = LSTM(10, 20, bidirectional=True)
    gru = GRU(10, 20, bias=False)
    assert rnn.weight_ih_l0.shape == (20, 10)
    assert lstm.weight_ih_l0.shape == (40, 10)
    assert not hasattr(gru, "bias_ih_l0")
    builder = GraphBuilder("dummy")
    with Tracing(builder):
        x = Tensor((5, 3, 10), DType.FLOAT32, "x")
        y1, h1 = rnn(x)
        y2, h2 = lstm(x)
        y3, h3 = gru(x)
    assert y1 is not None


def test_rnn_list_return():
    """Provides semantic functionality and verification."""
    from unittest.mock import patch
    from onnx9000.frontends.frontend.nn.rnn import RNN, LSTM, GRU

    rnn = RNN(10, 20)
    lstm = LSTM(10, 20, bidirectional=True)
    gru = GRU(10, 20, bias=False)
    with patch(
        "onnx9000.frontends.frontend.utils.record_op", return_value=["a", "b", "c"]
    ):
        y1, h1 = rnn(None, hx="dummy_hx")
        y2, h2 = lstm(None, hx=("d1", "d2"))
        y3, h3 = gru(None, hx="dummy_hx")
        assert y1 == "a"
        assert h1 == "b"
        assert y2 == "a"
        assert h2 == ("b", "c")
        assert y3 == "a"
        assert h3 == "b"
