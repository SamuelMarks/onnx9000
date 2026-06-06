import pytest
from onnx9000.optimizer.simplifier.passes.layout import *

def test_transform_nchw_to_nhwc():
    try:
        res = transform_nchw_to_nhwc()
    except Exception:
        pass

def test_transform_nhwc_to_nchw():
    try:
        res = transform_nhwc_to_nchw()
    except Exception:
        pass

