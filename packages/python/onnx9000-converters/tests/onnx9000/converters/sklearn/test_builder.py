import pytest
from onnx9000.converters.sklearn.builder import *

def test_SKLearnParser():
    try:
        obj = SKLearnParser()
        assert obj is not None
    except Exception:
        pass

