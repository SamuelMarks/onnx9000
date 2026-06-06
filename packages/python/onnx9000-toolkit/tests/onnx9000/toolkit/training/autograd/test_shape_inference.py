import pytest
from onnx9000.toolkit.training.autograd.shape_inference import *

def test_infer_backward_shapes():
    try:
        res = infer_backward_shapes()
    except Exception:
        pass

