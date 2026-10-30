"""Ops conv."""

import ctypes

from onnx9000.core.exceptions import BackendExecutionError
from onnx9000.tensorrt.ffi import ffi
from onnx9000.tensorrt.network import INetworkDefinition, ITensor
from onnx9000.tensorrt.registry import register_op


@register_op("", "Conv")
def trt_conv(
    network: INetworkDefinition,
    node: int | float | str | bool | bytes | list | dict | tuple | None,
    tensors: dict[str, ITensor],
) -> None:
    """Execute trt_conv."""
    add_conv_func = getattr(ffi.lib, "addConvolutionNd", None)
    if not add_conv_func:
        raise BackendExecutionError("addConvolutionNd not found")

    in_name = node.inputs[0]
    node.inputs[1]

    input_tensor = tensors[in_name]

    # Needs to get weights and biases correctly from initializers.
    # We assume 'node.attributes' contains these for a zero-build demo.
    num_outputs = 1  # Example
    ctypes.pointer(ctypes.c_int32(3))  # Example Dims wrapper
    ctypes.c_void_p(0)
    ctypes.c_void_p(0)

    add_conv_func.restype = ctypes.c_void_p
    # The actual signature for addConvolutionNd:
    # ITensor* addConvolutionNd(ITensor& input, int32_t nbOutputMaps, Dims kernelSize, Weights kernelWeights, Weights biasWeights)

    ptr = add_conv_func(
        ctypes.c_void_p(network.ptr),
        ctypes.c_void_p(input_tensor.ptr),
        ctypes.c_int32(num_outputs),
        ctypes.c_void_p(0),  # Dims
        ctypes.c_void_p(0),
        ctypes.c_void_p(0),
    )  # Weights

    if not ptr:
        raise BackendExecutionError("Failed to add Conv")

    out_name = node.outputs[0]
    tensors[out_name] = ITensor(ptr, out_name)
