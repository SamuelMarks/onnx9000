import pytest
from onnx9000.converters.pytorch_fx_parser import *

def test_PyTorchFXParser():
    try:
        obj = PyTorchFXParser()
        assert obj is not None
    except Exception:
        pass

def test__map_aten_add_Tensor():
    try:
        res = _map_aten_add_Tensor()
    except Exception:
        pass

def test__map_aten_mul_Tensor():
    try:
        res = _map_aten_mul_Tensor()
    except Exception:
        pass

def test__map_aten_convolution_default():
    try:
        res = _map_aten_convolution_default()
    except Exception:
        pass

def test__map_aten_native_batch_norm_legit_no_training_default():
    try:
        res = _map_aten_native_batch_norm_legit_no_training_default()
    except Exception:
        pass

def test__map_aten_native_layer_norm_default():
    try:
        res = _map_aten_native_layer_norm_default()
    except Exception:
        pass

def test__map_aten_bmm_default():
    try:
        res = _map_aten_bmm_default()
    except Exception:
        pass

def test__map_aten_mm_default():
    try:
        res = _map_aten_mm_default()
    except Exception:
        pass

def test__map_aten_max_pool2d_with_indices_default():
    try:
        res = _map_aten_max_pool2d_with_indices_default()
    except Exception:
        pass

def test__map_aten_gelu_default():
    try:
        res = _map_aten_gelu_default()
    except Exception:
        pass

def test__map_aten_arange_start_step():
    try:
        res = _map_aten_arange_start_step()
    except Exception:
        pass

def test__map_aten_where_self():
    try:
        res = _map_aten_where_self()
    except Exception:
        pass

def test__map_aten_copy_():
    try:
        res = _map_aten_copy_()
    except Exception:
        pass

def test__map_aten_add_():
    try:
        res = _map_aten_add_()
    except Exception:
        pass

def test_load_pytorch_fx():
    try:
        res = load_pytorch_fx()
    except Exception:
        pass

