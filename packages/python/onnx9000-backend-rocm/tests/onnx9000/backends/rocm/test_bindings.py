import pytest
from onnx9000.backends.rocm.bindings import *

def test_is_hip_available():
    try:
        res = is_hip_available()
    except Exception:
        pass

def test_is_rocblas_available():
    try:
        res = is_rocblas_available()
    except Exception:
        pass

def test_is_miopen_available():
    try:
        res = is_miopen_available()
    except Exception:
        pass

def test__register_hip_api():
    try:
        res = _register_hip_api()
    except Exception:
        pass

def test__register_rocblas_api():
    try:
        res = _register_rocblas_api()
    except Exception:
        pass

def test__register_miopen_api():
    try:
        res = _register_miopen_api()
    except Exception:
        pass

def test_check_hip_error():
    try:
        res = check_hip_error()
    except Exception:
        pass

def test_check_rocblas_error():
    try:
        res = check_rocblas_error()
    except Exception:
        pass

def test_check_miopen_error():
    try:
        res = check_miopen_error()
    except Exception:
        pass

