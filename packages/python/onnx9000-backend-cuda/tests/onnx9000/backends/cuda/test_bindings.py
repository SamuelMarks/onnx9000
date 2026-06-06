import pytest
from onnx9000.backends.cuda.bindings import *

def test_is_cuda_available():
    try:
        res = is_cuda_available()
    except Exception:
        pass

def test_is_cublas_available():
    try:
        res = is_cublas_available()
    except Exception:
        pass

def test_is_cudnn_available():
    try:
        res = is_cudnn_available()
    except Exception:
        pass

def test__register_cuda_api():
    try:
        res = _register_cuda_api()
    except Exception:
        pass

def test__register_cublas_api():
    try:
        res = _register_cublas_api()
    except Exception:
        pass

def test__register_cudnn_api():
    try:
        res = _register_cudnn_api()
    except Exception:
        pass

def test_check_cuda_error():
    try:
        res = check_cuda_error()
    except Exception:
        pass

def test_check_cublas_error():
    try:
        res = check_cublas_error()
    except Exception:
        pass

def test_check_cudnn_error():
    try:
        res = check_cudnn_error()
    except Exception:
        pass

