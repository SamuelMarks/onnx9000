"""Apple Accelerate and Metal Bindings."""

import ctypes
import ctypes.util
import logging
from typing import Optional

logger = logging.getLogger(__name__)
_accelerate_path = "/System/Library/Frameworks/Accelerate.framework/Accelerate"
_accelerate_lib = None
try:
    _accelerate_lib = ctypes.CDLL(_accelerate_path)
except Exception as e:
    logger.debug(f"Accelerate library not found: {e}")
_metal_path = "/System/Library/Frameworks/Metal.framework/Metal"
_metal_lib = None
try:
    _metal_lib = ctypes.CDLL(_metal_path)
except Exception as e:
    logger.debug(f"Metal library not found: {e}")
_mps_path = "/System/Library/Frameworks/MetalPerformanceShaders.framework/MetalPerformanceShaders"
_mps_lib = None
try:
    _mps_lib = ctypes.CDLL(_mps_path)
except Exception as e:
    logger.debug(f"MPS library not found: {e}")
_foundation_path = "/System/Library/Frameworks/Foundation.framework/Foundation"
_foundation_lib = None
try:
    _foundation_lib = ctypes.CDLL(_foundation_path)
except Exception as e:
    logger.debug(f"Foundation library not found: {e}")


def is_accelerate_available() -> bool:
    """Executes the is accelerate available operation."""
    return _accelerate_lib is not None


def is_metal_available() -> bool:
    """Executes the is metal available operation."""
    return _metal_lib is not None


def is_mps_available() -> bool:
    """Executes the is mps available operation."""
    return _mps_lib is not None


if is_accelerate_available():
    _accelerate_lib.cblas_sgemm.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    _accelerate_lib.vDSP_vadd.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_size_t,
    ]
    _accelerate_lib.vDSP_vsub.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_size_t,
    ]
    _accelerate_lib.vDSP_vmul.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_size_t,
    ]
    _accelerate_lib.vDSP_vsmul.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_size_t,
    ]
_objc = ctypes.CDLL(ctypes.util.find_library("objc"))
_objc.objc_getClass.restype = ctypes.c_void_p
_objc.sel_registerName.restype = ctypes.c_void_p
_objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_objc.objc_msgSend.restype = ctypes.c_void_p


def get_class(name: str) -> ctypes.c_void_p:
    """Executes the get class operation."""
    return _objc.objc_getClass(name.encode("utf-8"))


def get_selector(name: str) -> ctypes.c_void_p:
    """Executes the get selector operation."""
    return _objc.sel_registerName(name.encode("utf-8"))


def nsstring(string: str) -> ctypes.c_void_p:
    """Create an NSString."""
    if not _foundation_lib:
        return ctypes.c_void_p(0)
    NSString = get_class("NSString")
    stringWithUTF8String = get_selector("stringWithUTF8String:")
    msg_send_str = ctypes.cast(
        _objc.objc_msgSend,
        ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p),
    )
    return msg_send_str(NSString, stringWithUTF8String, string.encode("utf-8"))


def mtl_create_system_default_device() -> Optional[ctypes.c_void_p]:
    """MTLCreateSystemDefaultDevice"""
    if not is_metal_available():
        return None
    try:
        func = _metal_lib.MTLCreateSystemDefaultDevice
        func.restype = ctypes.c_void_p
        return func()
    except AttributeError:
        return None
