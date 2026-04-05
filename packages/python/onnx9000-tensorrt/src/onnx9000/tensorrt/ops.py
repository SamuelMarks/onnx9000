"""Ops."""

import ctypes
from typing import Any

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


def _get_input(node, tensors, idx):
    """Get input."""
    if len(node.inputs) > idx:
        return tensors[node.inputs[idx]]
    return None


@register_op("", "Add")
def trt_add(network: INetworkDefinition, node: Any, tensors: dict[str, ITensor]):
    """Execute trt_add."""
    add_elementwise_func = getattr(ffi.lib, "addElementWise", None)
    if not add_elementwise_func:
        raise RuntimeError("addElementWise not found")
    in1 = _get_input(node, tensors, 0)
    in2 = _get_input(node, tensors, 1)
    if in1 is None or in2 is None:
        raise RuntimeError("Missing inputs for Add")
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
        ctypes.c_int32(ElementWiseOperation.kSUM.value),
    )
    if not ptr:
        raise RuntimeError("Failed to add Add")
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


@register_op("", "Sub")
def trt_sub(network: INetworkDefinition, node: Any, tensors: dict[str, ITensor]):
    """Execute trt_sub."""
    add_elementwise_func = getattr(ffi.lib, "addElementWise", None)
    if not add_elementwise_func:
        raise RuntimeError("addElementWise not found")
    in1 = _get_input(node, tensors, 0)
    in2 = _get_input(node, tensors, 1)
    if in1 is None or in2 is None:
        raise RuntimeError("Missing inputs for Sub")
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
        ctypes.c_int32(ElementWiseOperation.kSUB.value),
    )
    if not ptr:
        raise RuntimeError("Failed to add Sub")
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


@register_op("", "Mul")
def trt_mul(network: INetworkDefinition, node: Any, tensors: dict[str, ITensor]):
    """Execute trt_mul."""
    add_elementwise_func = getattr(ffi.lib, "addElementWise", None)
    if not add_elementwise_func:
        raise RuntimeError("addElementWise not found")
    in1 = _get_input(node, tensors, 0)
    in2 = _get_input(node, tensors, 1)
    if in1 is None or in2 is None:
        raise RuntimeError("Missing inputs for Mul")
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
        ctypes.c_int32(ElementWiseOperation.kPROD.value),
    )
    if not ptr:
        raise RuntimeError("Failed to add Mul")
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


@register_op("", "Div")
def trt_div(network: INetworkDefinition, node: Any, tensors: dict[str, ITensor]):
    """Execute trt_div."""
    add_elementwise_func = getattr(ffi.lib, "addElementWise", None)
    if not add_elementwise_func:
        raise RuntimeError("addElementWise not found")
    in1 = _get_input(node, tensors, 0)
    in2 = _get_input(node, tensors, 1)
    if in1 is None or in2 is None:
        raise RuntimeError("Missing inputs for Div")
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
        ctypes.c_int32(ElementWiseOperation.kDIV.value),
    )
    if not ptr:
        raise RuntimeError("Failed to add Div")
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


@register_op("", "Max")
def trt_max(network: INetworkDefinition, node: Any, tensors: dict[str, ITensor]):
    """Execute trt_max."""
    add_elementwise_func = getattr(ffi.lib, "addElementWise", None)
    if not add_elementwise_func:
        raise RuntimeError("addElementWise not found")
    in1 = _get_input(node, tensors, 0)
    in2 = _get_input(node, tensors, 1)
    if in1 is None or in2 is None:
        raise RuntimeError("Missing inputs for Max")
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
        ctypes.c_int32(ElementWiseOperation.kMAX.value),
    )
    if not ptr:
        raise RuntimeError("Failed to add Max")
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


@register_op("", "Min")
def trt_min(network: INetworkDefinition, node: Any, tensors: dict[str, ITensor]):
    """Execute trt_min."""
    add_elementwise_func = getattr(ffi.lib, "addElementWise", None)
    if not add_elementwise_func:
        raise RuntimeError("addElementWise not found")
    in1 = _get_input(node, tensors, 0)
    in2 = _get_input(node, tensors, 1)
    if in1 is None or in2 is None:
        raise RuntimeError("Missing inputs for Min")
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
        ctypes.c_int32(ElementWiseOperation.kMIN.value),
    )
    if not ptr:
        raise RuntimeError("Failed to add Min")
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


@register_op("", "Pow")
def trt_pow(network: INetworkDefinition, node: Any, tensors: dict[str, ITensor]):
    """Execute trt_pow."""
    add_elementwise_func = getattr(ffi.lib, "addElementWise", None)
    if not add_elementwise_func:
        raise RuntimeError("addElementWise not found")
    in1 = _get_input(node, tensors, 0)
    in2 = _get_input(node, tensors, 1)
    if in1 is None or in2 is None:
        raise RuntimeError("Missing inputs for Pow")
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
        ctypes.c_int32(ElementWiseOperation.kPOW.value),
    )
    if not ptr:
        raise RuntimeError("Failed to add Pow")
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


@register_op("", "Equal")
def trt_equal(network: INetworkDefinition, node: Any, tensors: dict[str, ITensor]):
    """Execute trt_equal."""
    add_elementwise_func = getattr(ffi.lib, "addElementWise", None)
    if not add_elementwise_func:
        raise RuntimeError("addElementWise not found")
    in1 = _get_input(node, tensors, 0)
    in2 = _get_input(node, tensors, 1)
    if in1 is None or in2 is None:
        raise RuntimeError("Missing inputs for Equal")
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
        ctypes.c_int32(ElementWiseOperation.kEQUAL.value),
    )
    if not ptr:
        raise RuntimeError("Failed to add Equal")
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


@register_op("", "Less")
def trt_less(network: INetworkDefinition, node: Any, tensors: dict[str, ITensor]):
    """Execute trt_less."""
    add_elementwise_func = getattr(ffi.lib, "addElementWise", None)
    if not add_elementwise_func:
        raise RuntimeError("addElementWise not found")
    in1 = _get_input(node, tensors, 0)
    in2 = _get_input(node, tensors, 1)
    if in1 is None or in2 is None:
        raise RuntimeError("Missing inputs for Less")
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
        ctypes.c_int32(ElementWiseOperation.kLESS.value),
    )
    if not ptr:
        raise RuntimeError("Failed to add Less")
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


@register_op("", "Greater")
def trt_greater(network: INetworkDefinition, node: Any, tensors: dict[str, ITensor]):
    """Execute trt_greater."""
    add_elementwise_func = getattr(ffi.lib, "addElementWise", None)
    if not add_elementwise_func:
        raise RuntimeError("addElementWise not found")
    in1 = _get_input(node, tensors, 0)
    in2 = _get_input(node, tensors, 1)
    if in1 is None or in2 is None:
        raise RuntimeError("Missing inputs for Greater")
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
        ctypes.c_int32(ElementWiseOperation.kGREATER.value),
    )
    if not ptr:
        raise RuntimeError("Failed to add Greater")
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


@register_op("", "And")
def trt_and(network: INetworkDefinition, node: Any, tensors: dict[str, ITensor]):
    """Execute trt_and."""
    add_elementwise_func = getattr(ffi.lib, "addElementWise", None)
    if not add_elementwise_func:
        raise RuntimeError("addElementWise not found")
    in1 = _get_input(node, tensors, 0)
    in2 = _get_input(node, tensors, 1)
    if in1 is None or in2 is None:
        raise RuntimeError("Missing inputs for And")
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
        ctypes.c_int32(ElementWiseOperation.kAND.value),
    )
    if not ptr:
        raise RuntimeError("Failed to add And")
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


@register_op("", "Or")
def trt_or(network: INetworkDefinition, node: Any, tensors: dict[str, ITensor]):
    """Execute trt_or."""
    add_elementwise_func = getattr(ffi.lib, "addElementWise", None)
    if not add_elementwise_func:
        raise RuntimeError("addElementWise not found")
    in1 = _get_input(node, tensors, 0)
    in2 = _get_input(node, tensors, 1)
    if in1 is None or in2 is None:
        raise RuntimeError("Missing inputs for Or")
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
        ctypes.c_int32(ElementWiseOperation.kOR.value),
    )
    if not ptr:
        raise RuntimeError("Failed to add Or")
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


@register_op("", "Xor")
def trt_xor(network: INetworkDefinition, node: Any, tensors: dict[str, ITensor]):
    """Execute trt_xor."""
    add_elementwise_func = getattr(ffi.lib, "addElementWise", None)
    if not add_elementwise_func:
        raise RuntimeError("addElementWise not found")
    in1 = _get_input(node, tensors, 0)
    in2 = _get_input(node, tensors, 1)
    if in1 is None or in2 is None:
        raise RuntimeError("Missing inputs for Xor")
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
        ctypes.c_int32(ElementWiseOperation.kXOR.value),
    )
    if not ptr:
        raise RuntimeError("Failed to add Xor")
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


@register_op("", "Exp")
def trt_exp(network: INetworkDefinition, node: Any, tensors: dict[str, ITensor]):
    """Execute trt_exp."""
    add_unary_func = getattr(ffi.lib, "addUnaryOperation", None)
    if not add_unary_func:
        raise RuntimeError("addUnaryOperation not found")
    in1 = _get_input(node, tensors, 0)
    if in1 is None:
        raise RuntimeError("Missing inputs for Exp")
    add_unary_func.restype = ctypes.c_void_p
    add_unary_func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
    ptr = add_unary_func(
        ctypes.c_void_p(network.ptr),
        ctypes.c_void_p(in1.ptr),
        ctypes.c_int32(UnaryOperation.kEXP.value),
    )
    if not ptr:
        raise RuntimeError("Failed to add Exp")
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


@register_op("", "Log")
def trt_log(network: INetworkDefinition, node: Any, tensors: dict[str, ITensor]):
    """Execute trt_log."""
    add_unary_func = getattr(ffi.lib, "addUnaryOperation", None)
    if not add_unary_func:
        raise RuntimeError("addUnaryOperation not found")
    in1 = _get_input(node, tensors, 0)
    if in1 is None:
        raise RuntimeError("Missing inputs for Log")
    add_unary_func.restype = ctypes.c_void_p
    add_unary_func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
    ptr = add_unary_func(
        ctypes.c_void_p(network.ptr),
        ctypes.c_void_p(in1.ptr),
        ctypes.c_int32(UnaryOperation.kLOG.value),
    )
    if not ptr:
        raise RuntimeError("Failed to add Log")
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


@register_op("", "Sqrt")
def trt_sqrt(network: INetworkDefinition, node: Any, tensors: dict[str, ITensor]):
    """Execute trt_sqrt."""
    add_unary_func = getattr(ffi.lib, "addUnaryOperation", None)
    if not add_unary_func:
        raise RuntimeError("addUnaryOperation not found")
    in1 = _get_input(node, tensors, 0)
    if in1 is None:
        raise RuntimeError("Missing inputs for Sqrt")
    add_unary_func.restype = ctypes.c_void_p
    add_unary_func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
    ptr = add_unary_func(
        ctypes.c_void_p(network.ptr),
        ctypes.c_void_p(in1.ptr),
        ctypes.c_int32(UnaryOperation.kSQRT.value),
    )
    if not ptr:
        raise RuntimeError("Failed to add Sqrt")
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


@register_op("", "Abs")
def trt_abs(network: INetworkDefinition, node: Any, tensors: dict[str, ITensor]):
    """Execute trt_abs."""
    add_unary_func = getattr(ffi.lib, "addUnaryOperation", None)
    if not add_unary_func:
        raise RuntimeError("addUnaryOperation not found")
    in1 = _get_input(node, tensors, 0)
    if in1 is None:
        raise RuntimeError("Missing inputs for Abs")
    add_unary_func.restype = ctypes.c_void_p
    add_unary_func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
    ptr = add_unary_func(
        ctypes.c_void_p(network.ptr),
        ctypes.c_void_p(in1.ptr),
        ctypes.c_int32(UnaryOperation.kABS.value),
    )
    if not ptr:
        raise RuntimeError("Failed to add Abs")
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


@register_op("", "Neg")
def trt_neg(network: INetworkDefinition, node: Any, tensors: dict[str, ITensor]):
    """Execute trt_neg."""
    add_unary_func = getattr(ffi.lib, "addUnaryOperation", None)
    if not add_unary_func:
        raise RuntimeError("addUnaryOperation not found")
    in1 = _get_input(node, tensors, 0)
    if in1 is None:
        raise RuntimeError("Missing inputs for Neg")
    add_unary_func.restype = ctypes.c_void_p
    add_unary_func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
    ptr = add_unary_func(
        ctypes.c_void_p(network.ptr),
        ctypes.c_void_p(in1.ptr),
        ctypes.c_int32(UnaryOperation.kNEG.value),
    )
    if not ptr:
        raise RuntimeError("Failed to add Neg")
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


@register_op("", "Not")
def trt_not(network: INetworkDefinition, node: Any, tensors: dict[str, ITensor]):
    """Execute trt_not."""
    add_unary_func = getattr(ffi.lib, "addUnaryOperation", None)
    if not add_unary_func:
        raise RuntimeError("addUnaryOperation not found")
    in1 = _get_input(node, tensors, 0)
    if in1 is None:
        raise RuntimeError("Missing inputs for Not")
    add_unary_func.restype = ctypes.c_void_p
    add_unary_func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
    ptr = add_unary_func(
        ctypes.c_void_p(network.ptr),
        ctypes.c_void_p(in1.ptr),
        ctypes.c_int32(UnaryOperation.kNOT.value),
    )
    if not ptr:
        raise RuntimeError("Failed to add Not")
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


@register_op("", "Relu")
def trt_relu(network: INetworkDefinition, node: Any, tensors: dict[str, ITensor]):
    """Execute trt_relu."""
    add_act_func = getattr(ffi.lib, "addActivation", None)
    if not add_act_func:
        raise RuntimeError("addActivation not found")
    in1 = _get_input(node, tensors, 0)
    if in1 is None:
        raise RuntimeError("Missing inputs for Relu")
    add_act_func.restype = ctypes.c_void_p
    add_act_func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
    ptr = add_act_func(
        ctypes.c_void_p(network.ptr),
        ctypes.c_void_p(in1.ptr),
        ctypes.c_int32(ActivationType.kRELU.value),
    )
    if not ptr:
        raise RuntimeError("Failed to add Relu")
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


@register_op("", "Sigmoid")
def trt_sigmoid(network: INetworkDefinition, node: Any, tensors: dict[str, ITensor]):
    """Execute trt_sigmoid."""
    add_act_func = getattr(ffi.lib, "addActivation", None)
    if not add_act_func:
        raise RuntimeError("addActivation not found")
    in1 = _get_input(node, tensors, 0)
    if in1 is None:
        raise RuntimeError("Missing inputs for Sigmoid")
    add_act_func.restype = ctypes.c_void_p
    add_act_func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
    ptr = add_act_func(
        ctypes.c_void_p(network.ptr),
        ctypes.c_void_p(in1.ptr),
        ctypes.c_int32(ActivationType.kSIGMOID.value),
    )
    if not ptr:
        raise RuntimeError("Failed to add Sigmoid")
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


@register_op("", "Tanh")
def trt_tanh(network: INetworkDefinition, node: Any, tensors: dict[str, ITensor]):
    """Execute trt_tanh."""
    add_act_func = getattr(ffi.lib, "addActivation", None)
    if not add_act_func:
        raise RuntimeError("addActivation not found")
    in1 = _get_input(node, tensors, 0)
    if in1 is None:
        raise RuntimeError("Missing inputs for Tanh")
    add_act_func.restype = ctypes.c_void_p
    add_act_func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
    ptr = add_act_func(
        ctypes.c_void_p(network.ptr),
        ctypes.c_void_p(in1.ptr),
        ctypes.c_int32(ActivationType.kTANH.value),
    )
    if not ptr:
        raise RuntimeError("Failed to add Tanh")
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


@register_op("", "LeakyRelu")
def trt_leakyrelu(network: INetworkDefinition, node: Any, tensors: dict[str, ITensor]):
    """Execute trt_leakyrelu."""
    add_act_func = getattr(ffi.lib, "addActivation", None)
    if not add_act_func:
        raise RuntimeError("addActivation not found")
    in1 = _get_input(node, tensors, 0)
    if in1 is None:
        raise RuntimeError("Missing inputs for LeakyRelu")
    add_act_func.restype = ctypes.c_void_p
    add_act_func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
    ptr = add_act_func(
        ctypes.c_void_p(network.ptr),
        ctypes.c_void_p(in1.ptr),
        ctypes.c_int32(ActivationType.kLEAKY_RELU.value),
    )
    if not ptr:
        raise RuntimeError("Failed to add LeakyRelu")
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


@register_op("", "Elu")
def trt_elu(network: INetworkDefinition, node: Any, tensors: dict[str, ITensor]):
    """Execute trt_elu."""
    add_act_func = getattr(ffi.lib, "addActivation", None)
    if not add_act_func:
        raise RuntimeError("addActivation not found")
    in1 = _get_input(node, tensors, 0)
    if in1 is None:
        raise RuntimeError("Missing inputs for Elu")
    add_act_func.restype = ctypes.c_void_p
    add_act_func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
    ptr = add_act_func(
        ctypes.c_void_p(network.ptr),
        ctypes.c_void_p(in1.ptr),
        ctypes.c_int32(ActivationType.kELU.value),
    )
    if not ptr:
        raise RuntimeError("Failed to add Elu")
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


@register_op("", "Selu")
def trt_selu(network: INetworkDefinition, node: Any, tensors: dict[str, ITensor]):
    """Execute trt_selu."""
    add_act_func = getattr(ffi.lib, "addActivation", None)
    if not add_act_func:
        raise RuntimeError("addActivation not found")
    in1 = _get_input(node, tensors, 0)
    if in1 is None:
        raise RuntimeError("Missing inputs for Selu")
    add_act_func.restype = ctypes.c_void_p
    add_act_func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
    ptr = add_act_func(
        ctypes.c_void_p(network.ptr),
        ctypes.c_void_p(in1.ptr),
        ctypes.c_int32(ActivationType.kSELU.value),
    )
    if not ptr:
        raise RuntimeError("Failed to add Selu")
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


@register_op("", "Softplus")
def trt_softplus(network: INetworkDefinition, node: Any, tensors: dict[str, ITensor]):
    """Execute trt_softplus."""
    add_act_func = getattr(ffi.lib, "addActivation", None)
    if not add_act_func:
        raise RuntimeError("addActivation not found")
    in1 = _get_input(node, tensors, 0)
    if in1 is None:
        raise RuntimeError("Missing inputs for Softplus")
    add_act_func.restype = ctypes.c_void_p
    add_act_func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
    ptr = add_act_func(
        ctypes.c_void_p(network.ptr),
        ctypes.c_void_p(in1.ptr),
        ctypes.c_int32(ActivationType.kSOFTPLUS.value),
    )
    if not ptr:
        raise RuntimeError("Failed to add Softplus")
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


@register_op("", "Clip")
def trt_clip(network: INetworkDefinition, node: Any, tensors: dict[str, ITensor]):
    """Execute trt_clip."""
    add_act_func = getattr(ffi.lib, "addActivation", None)
    if not add_act_func:
        raise RuntimeError("addActivation not found")
    in1 = _get_input(node, tensors, 0)
    if in1 is None:
        raise RuntimeError("Missing inputs for Clip")
    add_act_func.restype = ctypes.c_void_p
    add_act_func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
    ptr = add_act_func(
        ctypes.c_void_p(network.ptr),
        ctypes.c_void_p(in1.ptr),
        ctypes.c_int32(ActivationType.kCLIP.value),
    )
    if not ptr:
        raise RuntimeError("Failed to add Clip")
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


@register_op("", "HardSigmoid")
def trt_hardsigmoid(network: INetworkDefinition, node: Any, tensors: dict[str, ITensor]):
    """Execute trt_hardsigmoid."""
    add_act_func = getattr(ffi.lib, "addActivation", None)
    if not add_act_func:
        raise RuntimeError("addActivation not found")
    in1 = _get_input(node, tensors, 0)
    if in1 is None:
        raise RuntimeError("Missing inputs for HardSigmoid")
    add_act_func.restype = ctypes.c_void_p
    add_act_func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
    ptr = add_act_func(
        ctypes.c_void_p(network.ptr),
        ctypes.c_void_p(in1.ptr),
        ctypes.c_int32(ActivationType.kHARD_SIGMOID.value),
    )
    if not ptr:
        raise RuntimeError("Failed to add HardSigmoid")
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


@register_op("", "MaxPool")
def trt_maxpool(network: INetworkDefinition, node: Any, tensors: dict[str, ITensor]):
    """Execute trt_maxpool."""
    add_pool_func = getattr(ffi.lib, "addPoolingNd", None)
    if not add_pool_func:
        raise RuntimeError("addPoolingNd not found")
    in1 = _get_input(node, tensors, 0)
    if in1 is None:
        raise RuntimeError("Missing inputs for MaxPool")
    add_pool_func.restype = ctypes.c_void_p
    add_pool_func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]
    # Dims struct mock for kernel size
    from onnx9000.tensorrt.structs import Dims

    dims = Dims([1, 1])
    dims_ptr = ctypes.pointer(dims)
    ptr = add_pool_func(
        ctypes.c_void_p(network.ptr),
        ctypes.c_void_p(in1.ptr),
        ctypes.c_int32(PoolingType.kMAX.value),
        dims_ptr,
    )
    if not ptr:
        raise RuntimeError("Failed to add MaxPool")
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


@register_op("", "AveragePool")
def trt_averagepool(network: INetworkDefinition, node: Any, tensors: dict[str, ITensor]):
    """Execute trt_averagepool."""
    add_pool_func = getattr(ffi.lib, "addPoolingNd", None)
    if not add_pool_func:
        raise RuntimeError("addPoolingNd not found")
    in1 = _get_input(node, tensors, 0)
    if in1 is None:
        raise RuntimeError("Missing inputs for AveragePool")
    add_pool_func.restype = ctypes.c_void_p
    add_pool_func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]
    # Dims struct mock for kernel size
    from onnx9000.tensorrt.structs import Dims

    dims = Dims([1, 1])
    dims_ptr = ctypes.pointer(dims)
    ptr = add_pool_func(
        ctypes.c_void_p(network.ptr),
        ctypes.c_void_p(in1.ptr),
        ctypes.c_int32(PoolingType.kAVERAGE.value),
        dims_ptr,
    )
    if not ptr:
        raise RuntimeError("Failed to add AveragePool")
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


@register_op("", "ReduceMean")
def trt_reducemean(network: INetworkDefinition, node: Any, tensors: dict[str, ITensor]):
    """Execute trt_reducemean."""
    add_reduce_func = getattr(ffi.lib, "addReduce", None)
    if not add_reduce_func:
        raise RuntimeError("addReduce not found")
    in1 = _get_input(node, tensors, 0)
    if in1 is None:
        raise RuntimeError("Missing inputs for ReduceMean")
    add_reduce_func.restype = ctypes.c_void_p
    add_reduce_func.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_uint32,
        ctypes.c_bool,
    ]

    keep_dims = True
    if hasattr(node, "attributes") and "keepdims" in node.attributes:
        keep_dims = bool(node.attributes["keepdims"])

    axes = 0
    if hasattr(node, "attributes") and "axes" in node.attributes:
        for a in node.attributes["axes"]:
            axes |= 1 << a

    ptr = add_reduce_func(
        ctypes.c_void_p(network.ptr),
        ctypes.c_void_p(in1.ptr),
        ctypes.c_int32(ReduceOperation.kAVG.value),
        ctypes.c_uint32(axes),
        ctypes.c_bool(keep_dims),
    )
    if not ptr:
        raise RuntimeError("Failed to add ReduceMean")
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


@register_op("", "ReduceSum")
def trt_reducesum(network: INetworkDefinition, node: Any, tensors: dict[str, ITensor]):
    """Execute trt_reducesum."""
    add_reduce_func = getattr(ffi.lib, "addReduce", None)
    if not add_reduce_func:
        raise RuntimeError("addReduce not found")
    in1 = _get_input(node, tensors, 0)
    if in1 is None:
        raise RuntimeError("Missing inputs for ReduceSum")
    add_reduce_func.restype = ctypes.c_void_p
    add_reduce_func.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_uint32,
        ctypes.c_bool,
    ]

    keep_dims = True
    if hasattr(node, "attributes") and "keepdims" in node.attributes:
        keep_dims = bool(node.attributes["keepdims"])

    axes = 0
    if hasattr(node, "attributes") and "axes" in node.attributes:
        for a in node.attributes["axes"]:
            axes |= 1 << a

    ptr = add_reduce_func(
        ctypes.c_void_p(network.ptr),
        ctypes.c_void_p(in1.ptr),
        ctypes.c_int32(ReduceOperation.kSUM.value),
        ctypes.c_uint32(axes),
        ctypes.c_bool(keep_dims),
    )
    if not ptr:
        raise RuntimeError("Failed to add ReduceSum")
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


@register_op("", "ReduceMax")
def trt_reducemax(network: INetworkDefinition, node: Any, tensors: dict[str, ITensor]):
    """Execute trt_reducemax."""
    add_reduce_func = getattr(ffi.lib, "addReduce", None)
    if not add_reduce_func:
        raise RuntimeError("addReduce not found")
    in1 = _get_input(node, tensors, 0)
    if in1 is None:
        raise RuntimeError("Missing inputs for ReduceMax")
    add_reduce_func.restype = ctypes.c_void_p
    add_reduce_func.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_uint32,
        ctypes.c_bool,
    ]

    keep_dims = True
    if hasattr(node, "attributes") and "keepdims" in node.attributes:
        keep_dims = bool(node.attributes["keepdims"])

    axes = 0
    if hasattr(node, "attributes") and "axes" in node.attributes:
        for a in node.attributes["axes"]:
            axes |= 1 << a

    ptr = add_reduce_func(
        ctypes.c_void_p(network.ptr),
        ctypes.c_void_p(in1.ptr),
        ctypes.c_int32(ReduceOperation.kMAX.value),
        ctypes.c_uint32(axes),
        ctypes.c_bool(keep_dims),
    )
    if not ptr:
        raise RuntimeError("Failed to add ReduceMax")
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


@register_op("", "ReduceMin")
def trt_reducemin(network: INetworkDefinition, node: Any, tensors: dict[str, ITensor]):
    """Execute trt_reducemin."""
    add_reduce_func = getattr(ffi.lib, "addReduce", None)
    if not add_reduce_func:
        raise RuntimeError("addReduce not found")
    in1 = _get_input(node, tensors, 0)
    if in1 is None:
        raise RuntimeError("Missing inputs for ReduceMin")
    add_reduce_func.restype = ctypes.c_void_p
    add_reduce_func.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_uint32,
        ctypes.c_bool,
    ]

    keep_dims = True
    if hasattr(node, "attributes") and "keepdims" in node.attributes:
        keep_dims = bool(node.attributes["keepdims"])

    axes = 0
    if hasattr(node, "attributes") and "axes" in node.attributes:
        for a in node.attributes["axes"]:
            axes |= 1 << a

    ptr = add_reduce_func(
        ctypes.c_void_p(network.ptr),
        ctypes.c_void_p(in1.ptr),
        ctypes.c_int32(ReduceOperation.kMIN.value),
        ctypes.c_uint32(axes),
        ctypes.c_bool(keep_dims),
    )
    if not ptr:
        raise RuntimeError("Failed to add ReduceMin")
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])


@register_op("", "ReduceProd")
def trt_reduceprod(network: INetworkDefinition, node: Any, tensors: dict[str, ITensor]):
    """Execute trt_reduceprod."""
    add_reduce_func = getattr(ffi.lib, "addReduce", None)
    if not add_reduce_func:
        raise RuntimeError("addReduce not found")
    in1 = _get_input(node, tensors, 0)
    if in1 is None:
        raise RuntimeError("Missing inputs for ReduceProd")
    add_reduce_func.restype = ctypes.c_void_p
    add_reduce_func.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_uint32,
        ctypes.c_bool,
    ]

    keep_dims = True
    if hasattr(node, "attributes") and "keepdims" in node.attributes:
        keep_dims = bool(node.attributes["keepdims"])

    axes = 0
    if hasattr(node, "attributes") and "axes" in node.attributes:
        for a in node.attributes["axes"]:
            axes |= 1 << a

    ptr = add_reduce_func(
        ctypes.c_void_p(network.ptr),
        ctypes.c_void_p(in1.ptr),
        ctypes.c_int32(ReduceOperation.kPROD.value),
        ctypes.c_uint32(axes),
        ctypes.c_bool(keep_dims),
    )
    if not ptr:
        raise RuntimeError("Failed to add ReduceProd")
    tensors[node.outputs[0]] = ITensor(ptr, node.outputs[0])
