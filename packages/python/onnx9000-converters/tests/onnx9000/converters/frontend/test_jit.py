import pytest
from onnx9000.converters.frontend.jit import *

def test_jit():
    try:
        res = jit()
    except Exception:
        pass

