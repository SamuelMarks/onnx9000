"""TensorRT layer construction handlers for standard ONNX operators."""

import ctypes
from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol

from onnx9000.core.exceptions import BackendExecutionError
from onnx9000.tensorrt.enums import (
    ActivationType,
    ElementWiseOperation,
    PoolingType,
    ReduceOperation,
    UnaryOperation,
)
from onnx9000.tensorrt.ffi import ffi
from onnx9000.tensorrt.network import INetworkDefinition, ITensor
from onnx9000.tensorrt.registry import register_op
from onnx9000.tensorrt.structs import Dims


class TRTNodeProtocol(Protocol):
    """Protocol representing an ONNX node lowered into a TensorRT network."""

    @property
    def inputs(self) -> Sequence[str]:
        """Return ordered input tensor names."""

    @property
    def outputs(self) -> Sequence[str]:
        """Return ordered output tensor names."""

    @property
    def attributes(self) -> Mapping[str, int | float | bool | Iterable[int]]:
        """Return node attribute mapping."""


def _get_input(node: TRTNodeProtocol, tensors: Mapping[str, ITensor], idx: int) -> ITensor | None:
    """Retrieve the `idx`-th input `ITensor` for `node` if present.

    Args:
        node: ONNX node being lowered.
        tensors: Mapping of tensor names to active `ITensor` handles.
        idx: Zero-based input index.

    Returns:
        The corresponding `ITensor` if available, or `None`.
    """
    if len(node.inputs) > idx:
        return tensors.get(node.inputs[idx])
    return None


def _run_elementwise_op(
    network: INetworkDefinition,
    node: TRTNodeProtocol,
    tensors: dict[str, ITensor],
    op_name: str,
    op_enum: ElementWiseOperation,
) -> None:
    """Add a binary elementwise layer to the TensorRT network.

    Args:
        network: Target TensorRT `INetworkDefinition`.
        node: ONNX node with two inputs and one output.
        tensors: Mutable dictionary of network tensors.
        op_name: Canonical ONNX operator name for error reporting.
        op_enum: TensorRT `ElementWiseOperation` enum variant.

    Raises:
        RuntimeError: If `addElementWise` is unavailable, inputs are missing, or FFI returns null.
    """
    add_elementwise_func = ffi.lib.addElementWise if ffi.lib is not None else None
    if not add_elementwise_func:
        raise BackendExecutionError("addElementWise not found")
    in1 = _get_input(node, tensors, 0)
    in2 = _get_input(node, tensors, 1)
    if in1 is None or in2 is None:
        raise BackendExecutionError(f"Missing inputs for {op_name}")
    add_elementwise_func.restype = ctypes.c_void_p
    add_elementwise_func.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
    ]
    ptr = add_elementwise_func(
        ctypes.c_void_p(network.ptr),
        ctypes.c_void_p(in1.ptr),
        ctypes.c_void_p(in2.ptr),
        ctypes.c_int32(op_enum.value),
    )
    if not ptr:
        raise BackendExecutionError(f"Failed to add {op_name}")
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


def _run_unary_op(
    network: INetworkDefinition,
    node: TRTNodeProtocol,
    tensors: dict[str, ITensor],
    op_name: str,
    op_enum: UnaryOperation,
) -> None:
    """Add a unary operation layer to the TensorRT network.

    Args:
        network: Target TensorRT `INetworkDefinition`.
        node: ONNX node with one input and one output.
        tensors: Mutable dictionary of network tensors.
        op_name: Canonical ONNX operator name for error reporting.
        op_enum: TensorRT `UnaryOperation` enum variant.

    Raises:
        RuntimeError: If `addUnaryOperation` is unavailable, input is missing, or FFI returns null.
    """
    add_unary_func = ffi.lib.addUnaryOperation if ffi.lib is not None else None
    if not add_unary_func:
        raise BackendExecutionError("addUnaryOperation not found")
    in1 = _get_input(node, tensors, 0)
    if in1 is None:
        raise BackendExecutionError(f"Missing inputs for {op_name}")
    add_unary_func.restype = ctypes.c_void_p
    add_unary_func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
    ptr = add_unary_func(
        ctypes.c_void_p(network.ptr),
        ctypes.c_void_p(in1.ptr),
        ctypes.c_int32(op_enum.value),
    )
    if not ptr:
        raise BackendExecutionError(f"Failed to add {op_name}")
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


def _run_activation_op(
    network: INetworkDefinition,
    node: TRTNodeProtocol,
    tensors: dict[str, ITensor],
    op_name: str,
    act_enum: ActivationType,
) -> None:
    """Add an activation layer to the TensorRT network.

    Args:
        network: Target TensorRT `INetworkDefinition`.
        node: ONNX node with one input and one output.
        tensors: Mutable dictionary of network tensors.
        op_name: Canonical ONNX operator name for error reporting.
        act_enum: TensorRT `ActivationType` enum variant.

    Raises:
        RuntimeError: If `addActivation` is unavailable, input is missing, or FFI returns null.
    """
    add_act_func = ffi.lib.addActivation if ffi.lib is not None else None
    if not add_act_func:
        raise BackendExecutionError("addActivation not found")
    in1 = _get_input(node, tensors, 0)
    if in1 is None:
        raise BackendExecutionError(f"Missing inputs for {op_name}")
    add_act_func.restype = ctypes.c_void_p
    add_act_func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
    ptr = add_act_func(
        ctypes.c_void_p(network.ptr),
        ctypes.c_void_p(in1.ptr),
        ctypes.c_int32(act_enum.value),
    )
    if not ptr:
        raise BackendExecutionError(f"Failed to add {op_name}")
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


def _run_pool_op(
    network: INetworkDefinition,
    node: TRTNodeProtocol,
    tensors: dict[str, ITensor],
    op_name: str,
    pool_enum: PoolingType,
) -> None:
    """Add an N-dimensional pooling layer to the TensorRT network.

    Args:
        network: Target TensorRT `INetworkDefinition`.
        node: ONNX pooling node.
        tensors: Mutable dictionary of network tensors.
        op_name: Canonical ONNX operator name for error reporting.
        pool_enum: TensorRT `PoolingType` enum variant.

    Raises:
        RuntimeError: If `addPoolingNd` is unavailable, input is missing, or FFI returns null.
    """
    add_pool_func = ffi.lib.addPoolingNd if ffi.lib is not None else None
    if not add_pool_func:
        raise BackendExecutionError("addPoolingNd not found")
    in1 = _get_input(node, tensors, 0)
    if in1 is None:
        raise BackendExecutionError(f"Missing inputs for {op_name}")
    add_pool_func.restype = ctypes.c_void_p
    add_pool_func.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_void_p,
    ]
    dims = Dims([1, 1])
    dims_ptr = ctypes.pointer(dims)
    ptr = add_pool_func(
        ctypes.c_void_p(network.ptr),
        ctypes.c_void_p(in1.ptr),
        ctypes.c_int32(pool_enum.value),
        dims_ptr,
    )
    if not ptr:
        raise BackendExecutionError(f"Failed to add {op_name}")
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


def _run_reduce_op(
    network: INetworkDefinition,
    node: TRTNodeProtocol,
    tensors: dict[str, ITensor],
    op_name: str,
    reduce_enum: ReduceOperation,
) -> None:
    """Add a reduction layer to the TensorRT network.

    Args:
        network: Target TensorRT `INetworkDefinition`.
        node: ONNX reduction node.
        tensors: Mutable dictionary of network tensors.
        op_name: Canonical ONNX operator name for error reporting.
        reduce_enum: TensorRT `ReduceOperation` enum variant.

    Raises:
        RuntimeError: If `addReduce` is unavailable, input is missing, or FFI returns null.
    """
    add_reduce_func = ffi.lib.addReduce if ffi.lib is not None else None
    if not add_reduce_func:
        raise BackendExecutionError("addReduce not found")
    in1 = _get_input(node, tensors, 0)
    if in1 is None:
        raise BackendExecutionError(f"Missing inputs for {op_name}")
    add_reduce_func.restype = ctypes.c_void_p
    add_reduce_func.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_uint32,
        ctypes.c_bool,
    ]

    attrs = node.attributes
    keep_dims = bool(attrs.get("keepdims", True))

    axes = 0
    if "axes" in attrs:
        raw_axes = attrs["axes"]
        if isinstance(raw_axes, Iterable):
            for a in raw_axes:
                axes |= 1 << int(a)

    ptr = add_reduce_func(
        ctypes.c_void_p(network.ptr),
        ctypes.c_void_p(in1.ptr),
        ctypes.c_int32(reduce_enum.value),
        ctypes.c_uint32(axes),
        ctypes.c_bool(keep_dims),
    )
    if not ptr:
        raise BackendExecutionError(f"Failed to add {op_name}")
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


@register_op("", "Add")
def trt_add(
    network: INetworkDefinition, node: TRTNodeProtocol, tensors: dict[str, ITensor]
) -> None:
    """Lower an ONNX `Add` node into TensorRT.

    Args:
        network: Target TensorRT network definition.
        node: ONNX `Add` node.
        tensors: Active network tensor dictionary.
    """
    _run_elementwise_op(network, node, tensors, "Add", ElementWiseOperation.kSUM)


@register_op("", "Sub")
def trt_sub(
    network: INetworkDefinition, node: TRTNodeProtocol, tensors: dict[str, ITensor]
) -> None:
    """Lower an ONNX `Sub` node into TensorRT.

    Args:
        network: Target TensorRT network definition.
        node: ONNX `Sub` node.
        tensors: Active network tensor dictionary.
    """
    _run_elementwise_op(network, node, tensors, "Sub", ElementWiseOperation.kSUB)


@register_op("", "Mul")
def trt_mul(
    network: INetworkDefinition, node: TRTNodeProtocol, tensors: dict[str, ITensor]
) -> None:
    """Lower an ONNX `Mul` node into TensorRT.

    Args:
        network: Target TensorRT network definition.
        node: ONNX `Mul` node.
        tensors: Active network tensor dictionary.
    """
    _run_elementwise_op(network, node, tensors, "Mul", ElementWiseOperation.kPROD)


@register_op("", "Div")
def trt_div(
    network: INetworkDefinition, node: TRTNodeProtocol, tensors: dict[str, ITensor]
) -> None:
    """Lower an ONNX `Div` node into TensorRT.

    Args:
        network: Target TensorRT network definition.
        node: ONNX `Div` node.
        tensors: Active network tensor dictionary.
    """
    _run_elementwise_op(network, node, tensors, "Div", ElementWiseOperation.kDIV)


@register_op("", "Max")
def trt_max(
    network: INetworkDefinition, node: TRTNodeProtocol, tensors: dict[str, ITensor]
) -> None:
    """Lower an ONNX `Max` node into TensorRT.

    Args:
        network: Target TensorRT network definition.
        node: ONNX `Max` node.
        tensors: Active network tensor dictionary.
    """
    _run_elementwise_op(network, node, tensors, "Max", ElementWiseOperation.kMAX)


@register_op("", "Min")
def trt_min(
    network: INetworkDefinition, node: TRTNodeProtocol, tensors: dict[str, ITensor]
) -> None:
    """Lower an ONNX `Min` node into TensorRT.

    Args:
        network: Target TensorRT network definition.
        node: ONNX `Min` node.
        tensors: Active network tensor dictionary.
    """
    _run_elementwise_op(network, node, tensors, "Min", ElementWiseOperation.kMIN)


@register_op("", "Pow")
def trt_pow(
    network: INetworkDefinition, node: TRTNodeProtocol, tensors: dict[str, ITensor]
) -> None:
    """Lower an ONNX `Pow` node into TensorRT.

    Args:
        network: Target TensorRT network definition.
        node: ONNX `Pow` node.
        tensors: Active network tensor dictionary.
    """
    _run_elementwise_op(network, node, tensors, "Pow", ElementWiseOperation.kPOW)


@register_op("", "Equal")
def trt_equal(
    network: INetworkDefinition, node: TRTNodeProtocol, tensors: dict[str, ITensor]
) -> None:
    """Lower an ONNX `Equal` node into TensorRT.

    Args:
        network: Target TensorRT network definition.
        node: ONNX `Equal` node.
        tensors: Active network tensor dictionary.
    """
    _run_elementwise_op(network, node, tensors, "Equal", ElementWiseOperation.kEQUAL)


@register_op("", "Less")
def trt_less(
    network: INetworkDefinition, node: TRTNodeProtocol, tensors: dict[str, ITensor]
) -> None:
    """Lower an ONNX `Less` node into TensorRT.

    Args:
        network: Target TensorRT network definition.
        node: ONNX `Less` node.
        tensors: Active network tensor dictionary.
    """
    _run_elementwise_op(network, node, tensors, "Less", ElementWiseOperation.kLESS)


@register_op("", "Greater")
def trt_greater(
    network: INetworkDefinition, node: TRTNodeProtocol, tensors: dict[str, ITensor]
) -> None:
    """Lower an ONNX `Greater` node into TensorRT.

    Args:
        network: Target TensorRT network definition.
        node: ONNX `Greater` node.
        tensors: Active network tensor dictionary.
    """
    _run_elementwise_op(network, node, tensors, "Greater", ElementWiseOperation.kGREATER)


@register_op("", "And")
def trt_and(
    network: INetworkDefinition, node: TRTNodeProtocol, tensors: dict[str, ITensor]
) -> None:
    """Lower an ONNX `And` node into TensorRT.

    Args:
        network: Target TensorRT network definition.
        node: ONNX `And` node.
        tensors: Active network tensor dictionary.
    """
    _run_elementwise_op(network, node, tensors, "And", ElementWiseOperation.kAND)


@register_op("", "Or")
def trt_or(network: INetworkDefinition, node: TRTNodeProtocol, tensors: dict[str, ITensor]) -> None:
    """Lower an ONNX `Or` node into TensorRT.

    Args:
        network: Target TensorRT network definition.
        node: ONNX `Or` node.
        tensors: Active network tensor dictionary.
    """
    _run_elementwise_op(network, node, tensors, "Or", ElementWiseOperation.kOR)


@register_op("", "Xor")
def trt_xor(
    network: INetworkDefinition, node: TRTNodeProtocol, tensors: dict[str, ITensor]
) -> None:
    """Lower an ONNX `Xor` node into TensorRT.

    Args:
        network: Target TensorRT network definition.
        node: ONNX `Xor` node.
        tensors: Active network tensor dictionary.
    """
    _run_elementwise_op(network, node, tensors, "Xor", ElementWiseOperation.kXOR)


@register_op("", "Exp")
def trt_exp(
    network: INetworkDefinition, node: TRTNodeProtocol, tensors: dict[str, ITensor]
) -> None:
    """Lower an ONNX `Exp` node into TensorRT.

    Args:
        network: Target TensorRT network definition.
        node: ONNX `Exp` node.
        tensors: Active network tensor dictionary.
    """
    _run_unary_op(network, node, tensors, "Exp", UnaryOperation.kEXP)


@register_op("", "Log")
def trt_log(
    network: INetworkDefinition, node: TRTNodeProtocol, tensors: dict[str, ITensor]
) -> None:
    """Lower an ONNX `Log` node into TensorRT.

    Args:
        network: Target TensorRT network definition.
        node: ONNX `Log` node.
        tensors: Active network tensor dictionary.
    """
    _run_unary_op(network, node, tensors, "Log", UnaryOperation.kLOG)


@register_op("", "Sqrt")
def trt_sqrt(
    network: INetworkDefinition, node: TRTNodeProtocol, tensors: dict[str, ITensor]
) -> None:
    """Lower an ONNX `Sqrt` node into TensorRT.

    Args:
        network: Target TensorRT network definition.
        node: ONNX `Sqrt` node.
        tensors: Active network tensor dictionary.
    """
    _run_unary_op(network, node, tensors, "Sqrt", UnaryOperation.kSQRT)


@register_op("", "Abs")
def trt_abs(
    network: INetworkDefinition, node: TRTNodeProtocol, tensors: dict[str, ITensor]
) -> None:
    """Lower an ONNX `Abs` node into TensorRT.

    Args:
        network: Target TensorRT network definition.
        node: ONNX `Abs` node.
        tensors: Active network tensor dictionary.
    """
    _run_unary_op(network, node, tensors, "Abs", UnaryOperation.kABS)


@register_op("", "Neg")
def trt_neg(
    network: INetworkDefinition, node: TRTNodeProtocol, tensors: dict[str, ITensor]
) -> None:
    """Lower an ONNX `Neg` node into TensorRT.

    Args:
        network: Target TensorRT network definition.
        node: ONNX `Neg` node.
        tensors: Active network tensor dictionary.
    """
    _run_unary_op(network, node, tensors, "Neg", UnaryOperation.kNEG)


@register_op("", "Not")
def trt_not(
    network: INetworkDefinition, node: TRTNodeProtocol, tensors: dict[str, ITensor]
) -> None:
    """Lower an ONNX `Not` node into TensorRT.

    Args:
        network: Target TensorRT network definition.
        node: ONNX `Not` node.
        tensors: Active network tensor dictionary.
    """
    _run_unary_op(network, node, tensors, "Not", UnaryOperation.kNOT)


@register_op("", "Relu")
def trt_relu(
    network: INetworkDefinition, node: TRTNodeProtocol, tensors: dict[str, ITensor]
) -> None:
    """Lower an ONNX `Relu` node into TensorRT.

    Args:
        network: Target TensorRT network definition.
        node: ONNX `Relu` node.
        tensors: Active network tensor dictionary.
    """
    _run_activation_op(network, node, tensors, "Relu", ActivationType.kRELU)


@register_op("", "Sigmoid")
def trt_sigmoid(
    network: INetworkDefinition, node: TRTNodeProtocol, tensors: dict[str, ITensor]
) -> None:
    """Lower an ONNX `Sigmoid` node into TensorRT.

    Args:
        network: Target TensorRT network definition.
        node: ONNX `Sigmoid` node.
        tensors: Active network tensor dictionary.
    """
    _run_activation_op(network, node, tensors, "Sigmoid", ActivationType.kSIGMOID)


@register_op("", "Tanh")
def trt_tanh(
    network: INetworkDefinition, node: TRTNodeProtocol, tensors: dict[str, ITensor]
) -> None:
    """Lower an ONNX `Tanh` node into TensorRT.

    Args:
        network: Target TensorRT network definition.
        node: ONNX `Tanh` node.
        tensors: Active network tensor dictionary.
    """
    _run_activation_op(network, node, tensors, "Tanh", ActivationType.kTANH)


@register_op("", "LeakyRelu")
def trt_leakyrelu(
    network: INetworkDefinition, node: TRTNodeProtocol, tensors: dict[str, ITensor]
) -> None:
    """Lower an ONNX `LeakyRelu` node into TensorRT.

    Args:
        network: Target TensorRT network definition.
        node: ONNX `LeakyRelu` node.
        tensors: Active network tensor dictionary.
    """
    _run_activation_op(network, node, tensors, "LeakyRelu", ActivationType.kLEAKY_RELU)


@register_op("", "Elu")
def trt_elu(
    network: INetworkDefinition, node: TRTNodeProtocol, tensors: dict[str, ITensor]
) -> None:
    """Lower an ONNX `Elu` node into TensorRT.

    Args:
        network: Target TensorRT network definition.
        node: ONNX `Elu` node.
        tensors: Active network tensor dictionary.
    """
    _run_activation_op(network, node, tensors, "Elu", ActivationType.kELU)


@register_op("", "Selu")
def trt_selu(
    network: INetworkDefinition, node: TRTNodeProtocol, tensors: dict[str, ITensor]
) -> None:
    """Lower an ONNX `Selu` node into TensorRT.

    Args:
        network: Target TensorRT network definition.
        node: ONNX `Selu` node.
        tensors: Active network tensor dictionary.
    """
    _run_activation_op(network, node, tensors, "Selu", ActivationType.kSELU)


@register_op("", "Softplus")
def trt_softplus(
    network: INetworkDefinition, node: TRTNodeProtocol, tensors: dict[str, ITensor]
) -> None:
    """Lower an ONNX `Softplus` node into TensorRT.

    Args:
        network: Target TensorRT network definition.
        node: ONNX `Softplus` node.
        tensors: Active network tensor dictionary.
    """
    _run_activation_op(network, node, tensors, "Softplus", ActivationType.kSOFTPLUS)


@register_op("", "Clip")
def trt_clip(
    network: INetworkDefinition, node: TRTNodeProtocol, tensors: dict[str, ITensor]
) -> None:
    """Lower an ONNX `Clip` node into TensorRT.

    Args:
        network: Target TensorRT network definition.
        node: ONNX `Clip` node.
        tensors: Active network tensor dictionary.
    """
    _run_activation_op(network, node, tensors, "Clip", ActivationType.kCLIP)


@register_op("", "HardSigmoid")
def trt_hardsigmoid(
    network: INetworkDefinition, node: TRTNodeProtocol, tensors: dict[str, ITensor]
) -> None:
    """Lower an ONNX `HardSigmoid` node into TensorRT.

    Args:
        network: Target TensorRT network definition.
        node: ONNX `HardSigmoid` node.
        tensors: Active network tensor dictionary.
    """
    _run_activation_op(network, node, tensors, "HardSigmoid", ActivationType.kHARD_SIGMOID)


@register_op("", "MaxPool")
def trt_maxpool(
    network: INetworkDefinition, node: TRTNodeProtocol, tensors: dict[str, ITensor]
) -> None:
    """Lower an ONNX `MaxPool` node into TensorRT.

    Args:
        network: Target TensorRT network definition.
        node: ONNX `MaxPool` node.
        tensors: Active network tensor dictionary.
    """
    _run_pool_op(network, node, tensors, "MaxPool", PoolingType.kMAX)


@register_op("", "AveragePool")
def trt_averagepool(
    network: INetworkDefinition, node: TRTNodeProtocol, tensors: dict[str, ITensor]
) -> None:
    """Lower an ONNX `AveragePool` node into TensorRT.

    Args:
        network: Target TensorRT network definition.
        node: ONNX `AveragePool` node.
        tensors: Active network tensor dictionary.
    """
    _run_pool_op(network, node, tensors, "AveragePool", PoolingType.kAVERAGE)


@register_op("", "ReduceMean")
def trt_reducemean(
    network: INetworkDefinition, node: TRTNodeProtocol, tensors: dict[str, ITensor]
) -> None:
    """Lower an ONNX `ReduceMean` node into TensorRT.

    Args:
        network: Target TensorRT network definition.
        node: ONNX `ReduceMean` node.
        tensors: Active network tensor dictionary.
    """
    _run_reduce_op(network, node, tensors, "ReduceMean", ReduceOperation.kAVG)


@register_op("", "ReduceSum")
def trt_reducesum(
    network: INetworkDefinition, node: TRTNodeProtocol, tensors: dict[str, ITensor]
) -> None:
    """Lower an ONNX `ReduceSum` node into TensorRT.

    Args:
        network: Target TensorRT network definition.
        node: ONNX `ReduceSum` node.
        tensors: Active network tensor dictionary.
    """
    _run_reduce_op(network, node, tensors, "ReduceSum", ReduceOperation.kSUM)


@register_op("", "ReduceMax")
def trt_reducemax(
    network: INetworkDefinition, node: TRTNodeProtocol, tensors: dict[str, ITensor]
) -> None:
    """Lower an ONNX `ReduceMax` node into TensorRT.

    Args:
        network: Target TensorRT network definition.
        node: ONNX `ReduceMax` node.
        tensors: Active network tensor dictionary.
    """
    _run_reduce_op(network, node, tensors, "ReduceMax", ReduceOperation.kMAX)


@register_op("", "ReduceMin")
def trt_reducemin(
    network: INetworkDefinition, node: TRTNodeProtocol, tensors: dict[str, ITensor]
) -> None:
    """Lower an ONNX `ReduceMin` node into TensorRT.

    Args:
        network: Target TensorRT network definition.
        node: ONNX `ReduceMin` node.
        tensors: Active network tensor dictionary.
    """
    _run_reduce_op(network, node, tensors, "ReduceMin", ReduceOperation.kMIN)


@register_op("", "ReduceProd")
def trt_reduceprod(
    network: INetworkDefinition, node: TRTNodeProtocol, tensors: dict[str, ITensor]
) -> None:
    """Lower an ONNX `ReduceProd` node into TensorRT.

    Args:
        network: Target TensorRT network definition.
        node: ONNX `ReduceProd` node.
        tensors: Active network tensor dictionary.
    """
    _run_reduce_op(network, node, tensors, "ReduceProd", ReduceOperation.kPROD)
