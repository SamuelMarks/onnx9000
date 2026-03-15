import ctypes
import pytest
from unittest.mock import patch, MagicMock
from onnx9000.backends.cuda.bindings import (
    is_cuda_available,
    is_cublas_available,
    is_cudnn_available,
    check_cuda_error,
    check_cublas_error,
    check_cudnn_error,
)


def test_check_cuda_error():
    check_cuda_error(0)
    with pytest.raises(RuntimeError):
        check_cuda_error(1)


def test_check_cublas_error():
    check_cublas_error(0)
    with pytest.raises(RuntimeError):
        check_cublas_error(1)


def test_check_cudnn_error():
    check_cudnn_error(0)
    with pytest.raises(RuntimeError):
        check_cudnn_error(1)


def test_availability():
    assert isinstance(is_cuda_available(), bool)
    assert isinstance(is_cublas_available(), bool)
    assert isinstance(is_cudnn_available(), bool)


def test_register_apis():
    from onnx9000.backends.cuda.bindings import (
        _register_cuda_api,
        _register_cublas_api,
        _register_cudnn_api,
    )

    mock_lib = MagicMock()
    _register_cuda_api(mock_lib)
    assert hasattr(mock_lib.cuInit, "argtypes")
    mock_cublas = MagicMock()
    _register_cublas_api(mock_cublas)
    assert hasattr(mock_cublas.cublasCreate_v2, "argtypes")
    mock_cudnn = MagicMock()
    _register_cudnn_api(mock_cudnn)
    assert hasattr(mock_cudnn.cudnnCreate, "argtypes")
