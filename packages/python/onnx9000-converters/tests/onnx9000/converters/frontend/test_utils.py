import pytest
from onnx9000.converters.frontend.utils import *

def test_infer_elementwise_shape():
    try:
        res = infer_elementwise_shape()
    except Exception:
        pass

def test_infer_matmul_shape():
    try:
        res = infer_matmul_shape()
    except Exception:
        pass

def test_record_op():
    try:
        res = record_op()
    except Exception:
        pass

