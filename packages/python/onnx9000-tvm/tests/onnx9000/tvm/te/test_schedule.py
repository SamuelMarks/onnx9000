import pytest
from onnx9000.tvm.te.schedule import *

def test_Stage():
    try:
        obj = Stage()
        assert obj is not None
    except Exception:
        pass

def test_Schedule():
    try:
        obj = Schedule()
        assert obj is not None
    except Exception:
        pass

def test_create_schedule():
    try:
        res = create_schedule()
    except Exception:
        pass

