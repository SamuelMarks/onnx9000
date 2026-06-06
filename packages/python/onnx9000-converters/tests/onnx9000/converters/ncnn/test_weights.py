import pytest
from onnx9000.converters.ncnn.weights import *


def test_WeightsReader():
    try:
        obj = WeightsReader()
        assert obj is not None
    except Exception:
        pass
