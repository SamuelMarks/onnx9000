"""ROCm CTypes bindings."""

import ctypes
import ctypes.util
import logging
import os

logger = logging.getLogger(__name__)

# Try loading libraries
_hip_lib = None
_rocblas_lib = None
_miopen_lib = None

try:
    _hip_path = ctypes.util.find_library("amdhip64") or "libamdhip64.so"
    _hip_lib = ctypes.CDLL(_hip_path)
except Exception as e:
    logger.debug(f"HIP library not found: {e}")

try:
    _rocblas_path = ctypes.util.find_library("rocblas") or "librocblas.so"
    _rocblas_lib = ctypes.CDLL(_rocblas_path)
except Exception as e:
    logger.debug(f"rocBLAS library not found: {e}")

try:
    _miopen_path = ctypes.util.find_library("MIOpen") or "libMIOpen.so"
    _miopen_lib = ctypes.CDLL(_miopen_path)
except Exception as e:
    logger.debug(f"MIOpen library not found: {e}")


def is_hip_available() -> bool:
    """Executes the is hip available operation."""
    return _hip_lib is not None


def is_rocblas_available() -> bool:
    """Executes the is rocblas available operation."""
    return _rocblas_lib is not None


def is_miopen_available() -> bool:
    """Executes the is miopen available operation."""
    return _miopen_lib is not None


# Types
hipDeviceptr_t = ctypes.c_void_p
hipStream_t = ctypes.c_void_p
rocblas_handle = ctypes.c_void_p
miopenHandle_t = ctypes.c_void_p
miopenTensorDescriptor_t = ctypes.c_void_p


def _register_hip_api(lib):
    """Executes the  register hip api operation."""
    if lib is None:
        return
    lib.hipMalloc.argtypes = [ctypes.POINTER(hipDeviceptr_t), ctypes.c_size_t]
    lib.hipFree.argtypes = [hipDeviceptr_t]
    lib.hipMemcpyHtoD.argtypes = [hipDeviceptr_t, ctypes.c_void_p, ctypes.c_size_t]
    lib.hipMemcpyDtoH.argtypes = [ctypes.c_void_p, hipDeviceptr_t, ctypes.c_size_t]
    lib.hipStreamCreate.argtypes = [ctypes.POINTER(hipStream_t)]
    lib.hipStreamDestroy.argtypes = [hipStream_t]
    lib.hipModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
    lib.hipModuleGetFunction.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    lib.hipModuleLaunchKernel.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        hipStream_t,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    ]


def _register_rocblas_api(lib):
    """Executes the  register rocblas api operation."""
    if lib is None:
        return
    lib.rocblas_create_handle.argtypes = [ctypes.POINTER(rocblas_handle)]
    lib.rocblas_destroy_handle.argtypes = [rocblas_handle]
    lib.rocblas_set_stream.argtypes = [rocblas_handle, hipStream_t]
    lib.rocblas_sgemm.argtypes = [
        rocblas_handle,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        hipDeviceptr_t,
        ctypes.c_int,
        hipDeviceptr_t,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        hipDeviceptr_t,
        ctypes.c_int,
    ]
    lib.rocblas_hgemm.argtypes = [
        rocblas_handle,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint16),
        hipDeviceptr_t,
        ctypes.c_int,
        hipDeviceptr_t,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint16),
        hipDeviceptr_t,
        ctypes.c_int,
    ]


def _register_miopen_api(lib):
    """Executes the  register miopen api operation."""
    if lib is None:
        return
    lib.miopenCreate.argtypes = [ctypes.POINTER(miopenHandle_t)]
    lib.miopenDestroy.argtypes = [miopenHandle_t]
    lib.miopenConvolutionForward.argtypes = [
        miopenHandle_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
        hipDeviceptr_t,
        ctypes.c_void_p,
        hipDeviceptr_t,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        hipDeviceptr_t,
    ]
    lib.miopenPoolingForward.argtypes = [
        miopenHandle_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        hipDeviceptr_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
        hipDeviceptr_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
        hipDeviceptr_t,
    ]
    lib.miopenActivationForward.argtypes = [
        miopenHandle_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        hipDeviceptr_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
        hipDeviceptr_t,
    ]
    lib.miopenSoftmaxForward.argtypes = [
        miopenHandle_t,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        hipDeviceptr_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
        hipDeviceptr_t,
    ]


_register_hip_api(_hip_lib)
_register_rocblas_api(_rocblas_lib)
_register_miopen_api(_miopen_lib)


def check_hip_error(result: int) -> None:
    """Executes the check hip error operation."""
    if result != 0:
        raise RuntimeError(f"HIP Error: {result}")


def check_rocblas_error(result: int) -> None:
    """Executes the check rocblas error operation."""
    if result != 0:
        raise RuntimeError(f"rocBLAS Error: {result}")


def check_miopen_error(result: int) -> None:
    """Executes the check miopen error operation."""
    if result != 0:
        raise RuntimeError(f"MIOpen Error: {result}")
