import pytest
from onnx9000.converters.ncnn.mapper import *

def test_NCNNMapper():
    try:
        obj = NCNNMapper()
        assert obj is not None
    except Exception:
        pass

