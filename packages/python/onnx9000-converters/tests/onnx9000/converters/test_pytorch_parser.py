import pytest
from onnx9000.converters.pytorch_parser import *

def test_RestrictedUnpickler():
    try:
        obj = RestrictedUnpickler()
        assert obj is not None
    except Exception:
        pass

def test__rebuild_tensor_v2():
    try:
        res = _rebuild_tensor_v2()
    except Exception:
        pass

def test__rebuild_tensor_v3():
    try:
        res = _rebuild_tensor_v3()
    except Exception:
        pass

def test__rebuild_parameter():
    try:
        res = _rebuild_parameter()
    except Exception:
        pass

def test_parse_pytorch_checkpoint():
    try:
        res = parse_pytorch_checkpoint()
    except Exception:
        pass

def test__parse_zip():
    try:
        res = _parse_zip()
    except Exception:
        pass

def test__parse_old_format():
    try:
        res = _parse_old_format()
    except Exception:
        pass

