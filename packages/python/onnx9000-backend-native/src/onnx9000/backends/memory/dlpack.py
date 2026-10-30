"""Provide dlpack module functionality."""

import ctypes

from onnx9000.core.exceptions import BackendExecutionError


class DLDataType(ctypes.Structure):
    """Represents the D L Data Type class."""

    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class DLDevice(ctypes.Structure):
    """Represents the D L Device class."""

    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class DLTensor(ctypes.Structure):
    """Represents the D L Tensor class."""

    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class DLManagedTensor(ctypes.Structure):
    """Represents the D L Managed Tensor class."""

    __dummy__ = True


DLManagedTensor._fields_ = [
    ("dl_tensor", DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", ctypes.CFUNCTYPE(None, ctypes.POINTER(DLManagedTensor))),
]
kDLCPU = 1
kDLCUDA = 2
kDLMetal = 8
kDLROCM = 10
kDLCUDAManaged = 13


def from_dlpack(
    ext_tensor: int | float | str | bool | bytes | list | dict | tuple | None,
) -> tuple[ctypes.c_void_p, tuple, tuple | None, DLDataType, int]:
    """Consume PyTorch `torch.Tensor`, JAX `jax.Array`, TensorFlow `tf.Tensor` directly via DLPack (Zero-copy).

    Extract raw memory pointers (`data_ptr`) strictly natively.
    Extract memory strides strictly natively.
    Extract data types (`DLDataType` struct) strictly natively.
    """
    if not hasattr(ext_tensor, "__dlpack__"):
        raise BackendExecutionError("Object does not support DLPack protocol")
    capsule = ext_tensor.__dlpack__()
    ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.POINTER(DLManagedTensor)
    ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    managed_tensor_ptr = ctypes.pythonapi.PyCapsule_GetPointer(capsule, b"dltensor")
    if not managed_tensor_ptr:
        raise BackendExecutionError("Failed to extract DLManagedTensor pointer")
    dl_tensor = managed_tensor_ptr.contents.dl_tensor
    ndim = dl_tensor.ndim
    shape = tuple(dl_tensor.shape[i] for i in range(ndim))
    strides = tuple(dl_tensor.strides[i] for i in range(ndim)) if dl_tensor.strides else None
    if strides:
        expected_stride = 1
        is_c_contiguous = True
        for i in reversed(range(ndim)):
            if strides[i] != expected_stride:
                is_c_contiguous = False
                break
            expected_stride *= shape[i]
        if not is_c_contiguous:
            raise BackendExecutionError(
                "Tensor is not C_CONTIGUOUS. Auto-copy fallback not yet implemented in FFI boundaries."
            )
    data_ptr = ctypes.c_void_p(dl_tensor.data or 0)
    device_type = dl_tensor.device.device_type
    return (data_ptr, shape, strides, dl_tensor.dtype, device_type)


def from_numpy(
    array: int | float | str | bool | bytes | list | dict | tuple | None,
) -> tuple[ctypes.c_void_p, tuple, tuple | None, str]:
    """Consume NumPy `np.ndarray` directly via `__array_interface__` (Zero-copy)."""
    if not hasattr(array, "__array_interface__"):
        raise BackendExecutionError("Object does not support __array_interface__")
    interface = array.__array_interface__
    (data_ptr, _read_only) = interface["data"]
    shape = interface["shape"]
    strides = interface.get("strides")
    typestr = interface["typestr"]
    if strides:
        expected_stride = int(interface["typestr"][2:])
        is_c_contiguous = True
        for i in reversed(range(len(shape))):
            if strides[i] != expected_stride:
                is_c_contiguous = False
                break
            expected_stride *= shape[i]
        if not is_c_contiguous:
            raise BackendExecutionError("NumPy array is not C_CONTIGUOUS.")
    return (ctypes.c_void_p(data_ptr), shape, strides, typestr)
