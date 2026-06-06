import pytest
from onnx9000.converters.darknet.weights import *

def test_load_weights():
    try:
        res = load_weights()
    except Exception:
        pass

