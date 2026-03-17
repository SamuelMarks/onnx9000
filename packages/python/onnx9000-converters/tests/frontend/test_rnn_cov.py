"""Module providing core logic and structural definitions."""


def test_rnn_coverage_extra_2() -> None:
    """Tests the test_rnn_coverage_extra_2 functionality."""
    from unittest.mock import patch
    from onnx9000.converters.frontend.nn.rnn import GRU, LSTM, RNN

    rnn = RNN(10, 20)
    lstm = LSTM(10, 20, bias=True)
    gru = GRU(10, 20)
    with patch("onnx9000.converters.frontend.utils.record_op", return_value="res"):
        (y1, h1) = rnn(None)
        (y2, h2) = lstm(None)
        (y3, h3) = gru(None)
        assert y1 == "res"
        assert h1 == "res"
        assert y2 == "res"
        assert h2 == ("res", "res")
        assert y3 == "res"
        assert h3 == "res"


def test_rnn_list_return() -> None:
    """Tests the test_rnn_list_return functionality."""
    from unittest.mock import patch
    from onnx9000.converters.frontend.nn.rnn import GRU, LSTM, RNN

    rnn = RNN(10, 20)
    lstm = LSTM(10, 20, bias=False)
    gru = GRU(10, 20, bias=True)
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


def test_lstm_list_len_2() -> None:
    """Tests the test_lstm_list_len_2 functionality."""
    from unittest.mock import patch
    from onnx9000.converters.frontend.nn.rnn import LSTM

    lstm = LSTM(10, 20, bias=False)
    with patch("onnx9000.converters.frontend.utils.record_op", return_value=["a", "b"]):
        (y2, h2) = lstm(None)
        assert h2 == ("b", "b")
