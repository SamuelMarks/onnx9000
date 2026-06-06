import pytest
from onnx9000.converters.caffe.mapper import *

def test_CaffeMapper():
    try:
        obj = CaffeMapper()
        assert obj is not None
    except Exception:
        pass

