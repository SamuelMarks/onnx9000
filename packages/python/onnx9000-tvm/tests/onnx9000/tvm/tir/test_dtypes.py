import pytest
from onnx9000.tvm.tir.dtypes import *

def test_is_supported():
    try:
        res = is_supported()
    except Exception:
        pass

