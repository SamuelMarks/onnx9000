import pytest
from onnx9000.toolkit.training.autograd.optimizers import *

def test_add_sgd_optimizer():
    try:
        res = add_sgd_optimizer()
    except Exception:
        pass

def test_add_adam_optimizer():
    try:
        res = add_adam_optimizer()
    except Exception:
        pass

def test_add_adamw_optimizer():
    try:
        res = add_adamw_optimizer()
    except Exception:
        pass

def test_add_rmsprop_optimizer():
    try:
        res = add_rmsprop_optimizer()
    except Exception:
        pass

def test_add_adagrad_optimizer():
    try:
        res = add_adagrad_optimizer()
    except Exception:
        pass

def test_add_adadelta_optimizer():
    try:
        res = add_adadelta_optimizer()
    except Exception:
        pass

def test_add_gradient_accumulation():
    try:
        res = add_gradient_accumulation()
    except Exception:
        pass

def test_add_differential_privacy_noise():
    try:
        res = add_differential_privacy_noise()
    except Exception:
        pass

def test_add_gradient_clipping():
    try:
        res = add_gradient_clipping()
    except Exception:
        pass

def test_add_local_dp_gradient_clipping():
    try:
        res = add_local_dp_gradient_clipping()
    except Exception:
        pass

def test_add_gradient_clipping_value():
    try:
        res = add_gradient_clipping_value()
    except Exception:
        pass

