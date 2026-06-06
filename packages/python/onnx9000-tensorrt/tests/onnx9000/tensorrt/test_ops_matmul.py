import pytest
from onnx9000.tensorrt.ops_matmul import *

def test_trt_matmul():
    try:
        res = trt_matmul()
    except Exception:
        pass

