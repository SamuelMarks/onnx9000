"""Module providing core logic and structural definitions."""


def test_rnn_coverage_extra_2():
    """Provides semantic functionality and verification."""
    from onnx9000.frontends.frontend.nn.rnn import LSTM, GRU, RNN
    from unittest.mock import patch

    rnn = RNN(10, 20)
    lstm = LSTM(10, 20, bias=True)
    gru = GRU(10, 20)
    with patch("onnx9000.frontends.frontend.utils.record_op", return_value="res"):
        y1, h1 = rnn(None)
        y2, h2 = lstm(None)
        y3, h3 = gru(None)
        assert y1 == "res"
        assert h1 == "res"
        assert y2 == "res"
        assert h2 == ("res", "res")
        assert y3 == "res"
        assert h3 == "res"


def test_rnn_list_return():
    """Provides semantic functionality and verification."""
    from onnx9000.frontends.frontend.nn.rnn import RNN, LSTM, GRU
    from unittest.mock import patch

    rnn = RNN(10, 20)
    lstm = LSTM(10, 20, bias=False)
    gru = GRU(10, 20, bias=True)
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


def test_lstm_list_len_2():
    """Provides semantic functionality and verification."""
    from onnx9000.frontends.frontend.nn.rnn import LSTM
    from unittest.mock import patch

    lstm = LSTM(10, 20, bias=False)
    with patch("onnx9000.frontends.frontend.utils.record_op", return_value=["a", "b"]):
        y2, h2 = lstm(None)
        assert h2 == ("b", "b")
