import pytest
from onnx9000.converters.mxnet.weights import *

def test_load_params():
    try:
        res = load_params()
    except Exception:
        pass

