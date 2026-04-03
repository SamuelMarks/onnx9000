"""Module docstring."""

import ctypes
from typing import Any

from onnx9000.tensorrt.ffi import ffi
from onnx9000.tensorrt.network import INetworkDefinition, ITensor
from onnx9000.tensorrt.registry import register_op


def _get_input(node, tensors, idx):
    if len(node.inputs) > idx:
        return tensors[node.inputs[idx]]
    return None


@register_op("", "Reshape")
def trt_reshape(network: INetworkDefinition, node: Any, tensors: dict[str, ITensor]):
    """Execute trt_reshape."""
    add_shuffle_func = getattr(ffi.lib, "addShuffle", None)
    if not add_shuffle_func:
        raise RuntimeError("addShuffle not found")
    in1 = _get_input(node, tensors, 0)
    if in1 is None:
        raise RuntimeError("Missing input for Reshape")
    add_shuffle_func.restype = ctypes.c_void_p
    add_shuffle_func.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    ptr = add_shuffle_func(ctypes.c_void_p(network.ptr), ctypes.c_void_p(in1.ptr))
    # Requires setReshapeDimensions
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


@register_op("", "Transpose")
def trt_transpose(network: INetworkDefinition, node: Any, tensors: dict[str, ITensor]):
    """Execute trt_transpose."""
    add_shuffle_func = getattr(ffi.lib, "addShuffle", None)
    if not add_shuffle_func:
        raise RuntimeError("addShuffle not found")
    in1 = _get_input(node, tensors, 0)
    ptr = add_shuffle_func(ctypes.c_void_p(network.ptr), ctypes.c_void_p(in1.ptr))
    # Requires setFirstTranspose
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


@register_op("", "Concat")
def trt_concat(network: INetworkDefinition, node: Any, tensors: dict[str, ITensor]):
    """Execute trt_concat."""
    add_concat_func = getattr(ffi.lib, "addConcatenation", None)
    if not add_concat_func:
        raise RuntimeError("addConcatenation not found")
    in_tensors = [_get_input(node, tensors, i) for i in range(len(node.inputs))]
    # Pointer array
    ptr_array = (ctypes.c_void_p * len(in_tensors))(*[t.ptr for t in in_tensors if t])
    add_concat_func.restype = ctypes.c_void_p
    add_concat_func.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int32]
    ptr = add_concat_func(ctypes.c_void_p(network.ptr), ptr_array, ctypes.c_int32(len(in_tensors)))
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


@register_op("", "Slice")
def trt_slice(network: INetworkDefinition, node: Any, tensors: dict[str, ITensor]):
    """Execute trt_slice."""
    add_slice_func = getattr(ffi.lib, "addSlice", None)
    if not add_slice_func:
        raise RuntimeError("addSlice not found")
    in1 = _get_input(node, tensors, 0)
    # Dims starts, sizes, strides
    from onnx9000.tensorrt.structs import Dims

    dims = Dims([1, 1])
    dims_ptr = ctypes.pointer(dims)
    add_slice_func.restype = ctypes.c_void_p
    add_slice_func.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    ptr = add_slice_func(
        ctypes.c_void_p(network.ptr), ctypes.c_void_p(in1.ptr), dims_ptr, dims_ptr, dims_ptr
    )
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


@register_op("", "Gather")
def trt_gather(network: INetworkDefinition, node: Any, tensors: dict[str, ITensor]):
    """Execute trt_gather."""
    add_gather_func = getattr(ffi.lib, "addGather", None)
    if not add_gather_func:
        raise RuntimeError("addGather not found")
    in1 = _get_input(node, tensors, 0)
    in2 = _get_input(node, tensors, 1)
    axis = 0
    if hasattr(node, "attributes") and "axis" in node.attributes:
        axis = int(node.attributes["axis"])
    add_gather_func.restype = ctypes.c_void_p
    add_gather_func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
    ptr = add_gather_func(
        ctypes.c_void_p(network.ptr),
        ctypes.c_void_p(in1.ptr),
        ctypes.c_void_p(in2.ptr),
        ctypes.c_int32(axis),
    )
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])
