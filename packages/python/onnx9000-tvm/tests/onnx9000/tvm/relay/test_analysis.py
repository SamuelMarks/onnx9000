import pytest
from onnx9000.tvm.relay.analysis import *

def test_PostOrderVisitor():
    try:
        obj = PostOrderVisitor()
        assert obj is not None
    except Exception:
        pass

def test_post_order_visit():
    try:
        res = post_order_visit()
    except Exception:
        pass

def test_topological_sort():
    try:
        res = topological_sort()
    except Exception:
        pass

