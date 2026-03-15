"""CUDA CTypes bindings."""

import ctypes
import ctypes.util
import logging
import os

logger = logging.getLogger(__name__)

# Try loading libraries
_cuda_lib = None
_cublas_lib = None
_cudnn_lib = None

try:
    _cuda_path = ctypes.util.find_library("cuda") or "libcuda.so"
    _cuda_lib = ctypes.CDLL(_cuda_path)
except Exception as e:
    logger.debug(f"CUDA library not found: {e}")

try:
    _cublas_path = ctypes.util.find_library("cublas") or "libcublas.so"
    _cublas_lib = ctypes.CDLL(_cublas_path)
except Exception as e:
    logger.debug(f"cuBLAS library not found: {e}")

try:
    _cudnn_path = ctypes.util.find_library("cudnn") or "libcudnn.so"
    _cudnn_lib = ctypes.CDLL(_cudnn_path)
except Exception as e:
    logger.debug(f"cuDNN library not found: {e}")


def is_cuda_available() -> bool:
    """Executes the is cuda available operation."""
    return _cuda_lib is not None


def is_cublas_available() -> bool:
    """Executes the is cublas available operation."""
    return _cublas_lib is not None


def is_cudnn_available() -> bool:
    """Executes the is cudnn available operation."""
    return _cudnn_lib is not None


# Types
CUdeviceptr = ctypes.c_size_t
CUstream = ctypes.c_void_p
cublasHandle_t = ctypes.c_void_p
cudnnHandle_t = ctypes.c_void_p
cudnnTensorDescriptor_t = ctypes.c_void_p
cudnnFilterDescriptor_t = ctypes.c_void_p


def _register_cuda_api(lib):
    """Executes the  register cuda api operation."""
    if lib is None:
        return
    lib.cuInit.argtypes = [ctypes.c_uint]
    lib.cuMemAlloc_v2.argtypes = [ctypes.POINTER(CUdeviceptr), ctypes.c_size_t]
    lib.cuMemFree_v2.argtypes = [CUdeviceptr]
    lib.cuMemcpyHtoD_v2.argtypes = [CUdeviceptr, ctypes.c_void_p, ctypes.c_size_t]
    lib.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, CUdeviceptr, ctypes.c_size_t]
    lib.cuStreamCreate.argtypes = [ctypes.POINTER(CUstream), ctypes.c_uint]
    lib.cuStreamDestroy_v2.argtypes = [CUstream]
    lib.cuModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
    lib.cuModuleGetFunction.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    lib.cuLaunchKernel.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        CUstream,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    ]


def _register_cublas_api(lib):
    """Executes the  register cublas api operation."""
    if lib is None:
        return
    lib.cublasCreate_v2.argtypes = [ctypes.POINTER(cublasHandle_t)]
    lib.cublasDestroy_v2.argtypes = [cublasHandle_t]
    lib.cublasSetStream_v2.argtypes = [cublasHandle_t, CUstream]
    lib.cublasSgemm_v2.argtypes = [
        cublasHandle_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        CUdeviceptr,
        ctypes.c_int,
        CUdeviceptr,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        CUdeviceptr,
        ctypes.c_int,
    ]
    lib.cublasHgemm.argtypes = [
        cublasHandle_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint16),
        CUdeviceptr,
        ctypes.c_int,
        CUdeviceptr,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint16),
        CUdeviceptr,
        ctypes.c_int,
    ]
    lib.cublasSgemv_v2.argtypes = [
        cublasHandle_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        CUdeviceptr,
        ctypes.c_int,
        CUdeviceptr,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        CUdeviceptr,
        ctypes.c_int,
    ]


def _register_cudnn_api(lib):
    """Executes the  register cudnn api operation."""
    if lib is None:
        return
    lib.cudnnCreate.argtypes = [ctypes.POINTER(cudnnHandle_t)]
    lib.cudnnSetTensorNdDescriptor.argtypes = [
        cudnnTensorDescriptor_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.cudnnSetFilterNdDescriptor.argtypes = [
        cudnnFilterDescriptor_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.cudnnConvolutionForward.argtypes = [
        cudnnHandle_t,
        ctypes.c_void_p,
        cudnnTensorDescriptor_t,
        CUdeviceptr,
        cudnnFilterDescriptor_t,
        CUdeviceptr,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        cudnnTensorDescriptor_t,
        CUdeviceptr,
    ]
    lib.cudnnPoolingForward.argtypes = [
        cudnnHandle_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
        cudnnTensorDescriptor_t,
        CUdeviceptr,
        ctypes.c_void_p,
        cudnnTensorDescriptor_t,
        CUdeviceptr,
    ]
    lib.cudnnActivationForward.argtypes = [
        cudnnHandle_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
        cudnnTensorDescriptor_t,
        CUdeviceptr,
        ctypes.c_void_p,
        cudnnTensorDescriptor_t,
        CUdeviceptr,
    ]
    lib.cudnnSoftmaxForward.argtypes = [
        cudnnHandle_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        cudnnTensorDescriptor_t,
        CUdeviceptr,
        ctypes.c_void_p,
        cudnnTensorDescriptor_t,
        CUdeviceptr,
    ]
    lib.cudnnBatchNormalizationForwardInference.argtypes = [
        cudnnHandle_t,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        cudnnTensorDescriptor_t,
        CUdeviceptr,
        cudnnTensorDescriptor_t,
        CUdeviceptr,
        cudnnTensorDescriptor_t,
        CUdeviceptr,
        CUdeviceptr,
        CUdeviceptr,
        CUdeviceptr,
        ctypes.c_double,
    ]


_register_cuda_api(_cuda_lib)
_register_cublas_api(_cublas_lib)
_register_cudnn_api(_cudnn_lib)


def check_cuda_error(result: int) -> None:
    """Check CUDA error code."""
    if result != 0:
        raise RuntimeError(f"CUDA Error: {result}")


def check_cublas_error(result: int) -> None:
    """Check CUBLAS error code."""
    if result != 0:
        raise RuntimeError(f"CUBLAS Error: {result}")


def check_cudnn_error(result: int) -> None:
    """Check CUDNN error code."""
    if result != 0:
        raise RuntimeError(f"CUDNN Error: {result}")
