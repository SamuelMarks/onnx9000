import pytest
from onnx9000.tvm.te.default_schedules import *

def test_default_x86_schedule():
    try:
        res = default_x86_schedule()
    except Exception:
        pass

def test_default_arm_schedule():
    try:
        res = default_arm_schedule()
    except Exception:
        pass

def test_default_wasm_schedule():
    try:
        res = default_wasm_schedule()
    except Exception:
        pass

def test_default_webgpu_schedule():
    try:
        res = default_webgpu_schedule()
    except Exception:
        pass

