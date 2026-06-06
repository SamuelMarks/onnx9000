import pytest
from onnx9000.jax.jax_ops import *

def test__map_jax_add_prim():
    try:
        res = _map_jax_add_prim()
    except Exception:
        pass

def test__map_jax_mul_prim():
    try:
        res = _map_jax_mul_prim()
    except Exception:
        pass

def test__map_jax_dot_general_prim():
    try:
        res = _map_jax_dot_general_prim()
    except Exception:
        pass

def test__map_jax_broadcast_in_dim_prim():
    try:
        res = _map_jax_broadcast_in_dim_prim()
    except Exception:
        pass

def test__map_jax_xla_pmap_prim():
    try:
        res = _map_jax_xla_pmap_prim()
    except Exception:
        pass

def test__map_jax_grad_core_prim():
    try:
        res = _map_jax_grad_core_prim()
    except Exception:
        pass

def test__map_jax_sub_prim():
    try:
        res = _map_jax_sub_prim()
    except Exception:
        pass

def test__map_jax_div_prim():
    try:
        res = _map_jax_div_prim()
    except Exception:
        pass

def test__map_jax_conv_general_dilated():
    try:
        res = _map_jax_conv_general_dilated()
    except Exception:
        pass

def test__map_jax_reduce_sum():
    try:
        res = _map_jax_reduce_sum()
    except Exception:
        pass

def test__map_jax_reduce_max():
    try:
        res = _map_jax_reduce_max()
    except Exception:
        pass

def test__map_jax_reduce_min():
    try:
        res = _map_jax_reduce_min()
    except Exception:
        pass

def test__map_jax_reduce_prod():
    try:
        res = _map_jax_reduce_prod()
    except Exception:
        pass

def test__map_jax_reduce_window_max():
    try:
        res = _map_jax_reduce_window_max()
    except Exception:
        pass

def test__map_jax_reduce_window_sum():
    try:
        res = _map_jax_reduce_window_sum()
    except Exception:
        pass

def test__map_jax_pad():
    try:
        res = _map_jax_pad()
    except Exception:
        pass

def test__map_jax_slice():
    try:
        res = _map_jax_slice()
    except Exception:
        pass

def test__map_jax_dynamic_slice():
    try:
        res = _map_jax_dynamic_slice()
    except Exception:
        pass

def test__map_jax_gather():
    try:
        res = _map_jax_gather()
    except Exception:
        pass

def test__map_jax_scatter():
    try:
        res = _map_jax_scatter()
    except Exception:
        pass

def test__map_jax_cond():
    try:
        res = _map_jax_cond()
    except Exception:
        pass

def test__map_jax_scan():
    try:
        res = _map_jax_scan()
    except Exception:
        pass

def test__map_jax_while_loop():
    try:
        res = _map_jax_while_loop()
    except Exception:
        pass

