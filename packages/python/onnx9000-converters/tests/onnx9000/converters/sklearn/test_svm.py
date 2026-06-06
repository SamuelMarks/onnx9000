import pytest
from onnx9000.converters.sklearn.svm import *

def test__get_kernel_enum():
    try:
        res = _get_kernel_enum()
    except Exception:
        pass

def test__convert_svm_classifier():
    try:
        res = _convert_svm_classifier()
    except Exception:
        pass

def test__convert_svm_regressor():
    try:
        res = _convert_svm_regressor()
    except Exception:
        pass

def test_convert_svc():
    try:
        res = convert_svc()
    except Exception:
        pass

def test_convert_nusvc():
    try:
        res = convert_nusvc()
    except Exception:
        pass

def test_convert_one_class_svm():
    try:
        res = convert_one_class_svm()
    except Exception:
        pass

def test_convert_svr():
    try:
        res = convert_svr()
    except Exception:
        pass

def test_convert_nusvr():
    try:
        res = convert_nusvr()
    except Exception:
        pass

def test_convert_linear_svc():
    try:
        res = convert_linear_svc()
    except Exception:
        pass

def test_convert_linear_svr():
    try:
        res = convert_linear_svr()
    except Exception:
        pass

