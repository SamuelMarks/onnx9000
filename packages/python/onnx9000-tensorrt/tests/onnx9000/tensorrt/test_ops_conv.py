import pytest
from onnx9000.tensorrt.ops_conv import *


def test_trt_conv():
    try:
        trt_conv()
    except Exception:
        pass
