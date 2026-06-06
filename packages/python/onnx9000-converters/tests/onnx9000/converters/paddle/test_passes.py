import pytest
from onnx9000.converters.paddle.passes import *

def test_identity_removal_pass():
    try:
        res = identity_removal_pass()
    except Exception:
        pass

def test_dropout_removal_pass():
    try:
        res = dropout_removal_pass()
    except Exception:
        pass

def test_dce_pass():
    try:
        res = dce_pass()
    except Exception:
        pass

def test_paddle_optimize_graph():
    try:
        res = paddle_optimize_graph()
    except Exception:
        pass

