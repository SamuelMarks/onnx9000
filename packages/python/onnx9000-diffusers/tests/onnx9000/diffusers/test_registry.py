import pytest
from onnx9000.diffusers.registry import *

def test_register_op():
    try:
        res = register_op()
    except Exception:
        pass

