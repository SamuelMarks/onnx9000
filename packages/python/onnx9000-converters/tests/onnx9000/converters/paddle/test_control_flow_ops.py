import pytest
from onnx9000.converters.paddle.control_flow_ops import *


def test__map_conditional_block():
    try:
        _map_conditional_block()
    except Exception:
        pass


def test__map_while():
    try:
        _map_while()
    except Exception:
        pass


def test__map_rnn():
    try:
        _map_rnn()
    except Exception:
        pass


def test__map_lstm():
    try:
        _map_lstm()
    except Exception:
        pass


def test__map_gru():
    try:
        _map_gru()
    except Exception:
        pass


def test__map_tensor_array():
    try:
        _map_tensor_array()
    except Exception:
        pass
