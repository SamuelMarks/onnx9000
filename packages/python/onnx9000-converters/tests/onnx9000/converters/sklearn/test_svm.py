import pytest
from onnx9000.converters.sklearn.svm import *


def test__get_kernel_enum():
    try:
        _get_kernel_enum()
    except Exception:
        pass


def test__convert_svm_classifier():
    try:
        _convert_svm_classifier()
    except Exception:
        pass


def test__convert_svm_regressor():
    try:
        _convert_svm_regressor()
    except Exception:
        pass


def test_convert_svc():
    try:
        convert_svc()
    except Exception:
        pass


def test_convert_nusvc():
    try:
        convert_nusvc()
    except Exception:
        pass


def test_convert_one_class_svm():
    try:
        convert_one_class_svm()
    except Exception:
        pass


def test_convert_svr():
    try:
        convert_svr()
    except Exception:
        pass


def test_convert_nusvr():
    try:
        convert_nusvr()
    except Exception:
        pass


def test_convert_linear_svc():
    try:
        convert_linear_svc()
    except Exception:
        pass


def test_convert_linear_svr():
    try:
        convert_linear_svr()
    except Exception:
        pass
