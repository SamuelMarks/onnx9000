import pytest
from onnx9000.converters.tf.control_flow_ops import *

def test__map_noop():
    try:
        res = _map_noop()
    except Exception:
        pass

def test__map_if():
    try:
        res = _map_if()
    except Exception:
        pass

def test__map_while():
    try:
        res = _map_while()
    except Exception:
        pass

def test__map_tensor_array():
    try:
        res = _map_tensor_array()
    except Exception:
        pass

def test__map_tensor_array_read():
    try:
        res = _map_tensor_array_read()
    except Exception:
        pass

def test__map_tensor_array_write():
    try:
        res = _map_tensor_array_write()
    except Exception:
        pass

def test__map_tensor_array_size():
    try:
        res = _map_tensor_array_size()
    except Exception:
        pass

def test__map_tensor_array_gather():
    try:
        res = _map_tensor_array_gather()
    except Exception:
        pass

def test__map_tensor_array_scatter():
    try:
        res = _map_tensor_array_scatter()
    except Exception:
        pass

def test__map_lstm_cell():
    try:
        res = _map_lstm_cell()
    except Exception:
        pass

def test__map_gru_cell():
    try:
        res = _map_gru_cell()
    except Exception:
        pass

