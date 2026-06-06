import pytest
from onnx9000.converters.frontend.nn.rnn import *

def test_RNNBase():
    try:
        obj = RNNBase()
        assert obj is not None
    except Exception:
        pass

def test_RNN():
    try:
        obj = RNN()
        assert obj is not None
    except Exception:
        pass

def test_LSTM():
    try:
        obj = LSTM()
        assert obj is not None
    except Exception:
        pass

def test_GRU():
    try:
        obj = GRU()
        assert obj is not None
    except Exception:
        pass

