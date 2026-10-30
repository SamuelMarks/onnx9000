"""Ops matmul."""

import ctypes

from onnx9000.core.exceptions import BackendExecutionError
from onnx9000.tensorrt.enums import MatrixOperation
from onnx9000.tensorrt.ffi import ffi
from onnx9000.tensorrt.network import INetworkDefinition, ITensor
from onnx9000.tensorrt.registry import register_op


@register_op("", "MatMul")
def trt_matmul(
    network: INetworkDefinition,
    node: int | float | str | bool | bytes | list | dict | tuple | None,
    tensors: dict[str, ITensor],
) -> None:
    """Execute trt_matmul."""
    add_matmul_func = getattr(ffi.lib, "addMatrixMultiply", None)
    if not add_matmul_func:
        raise BackendExecutionError("addMatrixMultiply not found")

    in1_name = node.inputs[0]
    in2_name = node.inputs[1]

    in1 = tensors[in1_name]
    in2 = tensors[in2_name]

    # We will assume kNONE for now. A full ONNX implementation would check the shapes to see if kVECTOR is needed.
    opA = MatrixOperation.kNONE
    opB = MatrixOperation.kNONE

    add_matmul_func.restype = ctypes.c_void_p
    add_matmul_func.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.c_int32,
    ]

    ptr = add_matmul_func(
        ctypes.c_void_p(network.ptr),
        ctypes.c_void_p(in1.ptr),
        ctypes.c_int32(opA.value),
        ctypes.c_void_p(in2.ptr),
        ctypes.c_int32(opB.value),
    )
    if not ptr:
        raise BackendExecutionError("Failed to add MatMul")

    out_name = node.outputs[0]
    tensors[out_name] = ITensor(ptr, out_name)
