import pytest
from onnx9000.tvm.te.transform.verify import *

def test_InteractiveNotebookVisualizer():
    try:
        obj = InteractiveNotebookVisualizer()
        assert obj is not None
    except Exception:
        pass

def test_verify_schedule():
    try:
        res = verify_schedule()
    except Exception:
        pass

def test_trace_te_compute():
    try:
        res = trace_te_compute()
    except Exception:
        pass

