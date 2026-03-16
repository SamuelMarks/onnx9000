import ctypes
from unittest.mock import MagicMock
import pytest
from onnx9000.backends.cuda.bindings import (
    _register_cublas_api,
    _register_cuda_api,
    _register_cudnn_api,
    check_cublas_error,
    check_cuda_error,
    check_cudnn_error,
    is_cublas_available,
    is_cuda_available,
    is_cudnn_available,
)


def test_bindings_availability_and_signatures():
    assert isinstance(is_cuda_available(), bool)
    assert isinstance(is_cublas_available(), bool)
    assert isinstance(is_cudnn_available(), bool)
    mock_lib = MagicMock()
    _register_cuda_api(mock_lib)
    assert mock_lib.cuInit.argtypes == [ctypes.c_uint]
    _register_cublas_api(mock_lib)
    assert mock_lib.cublasCreate_v2.argtypes is not None
    _register_cudnn_api(mock_lib)
    assert mock_lib.cudnnCreate.argtypes is not None
    _register_cuda_api(None)
    _register_cublas_api(None)
    _register_cudnn_api(None)


def test_bindings_errors():
    with pytest.raises(RuntimeError, match="CUDA Error"):
        check_cuda_error(1)
    check_cuda_error(0)
    with pytest.raises(RuntimeError, match="CUBLAS Error"):
        check_cublas_error(1)
    check_cublas_error(0)
    with pytest.raises(RuntimeError, match="CUDNN Error"):
        check_cudnn_error(1)
    check_cudnn_error(0)
