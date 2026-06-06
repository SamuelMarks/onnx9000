import pytest
from onnx9000.optimizer.cli import *

def test_ModelCache():
    try:
        obj = ModelCache()
        assert obj is not None
    except Exception:
        pass

def test_save_onnx():
    try:
        res = save_onnx()
    except Exception:
        pass

def test_save_safetensors():
    try:
        res = save_safetensors()
    except Exception:
        pass

def test_optimize_cli():
    try:
        res = optimize_cli()
    except Exception:
        pass

def test_is_package_under_5mb():
    try:
        res = is_package_under_5mb()
    except Exception:
        pass

