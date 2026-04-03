"""Module docstring."""

import onnx9000.tensorrt.ops
import onnx9000.tensorrt.ops_conv
import onnx9000.tensorrt.ops_dim
import onnx9000.tensorrt.ops_matmul
from onnx9000.tensorrt.builder import Builder, BuilderConfig, NetworkDefinition
from onnx9000.tensorrt.enums import (
    ActivationType,
    BuilderFlag,
    DataType,
    ElementWiseOperation,
    MatrixOperation,
    MemoryPoolType,
    OptProfileSelector,
    PoolingType,
    ReduceOperation,
    ScaleMode,
    TopKOperation,
    UnaryOperation,
)
from onnx9000.tensorrt.ffi import TensorRTFFI, ffi
from onnx9000.tensorrt.network import INetworkDefinition, ITensor
from onnx9000.tensorrt.registry import get_op_translator, register_op
from onnx9000.tensorrt.structs import Dims, Weights

__all__ = [
    "ffi",
    "TensorRTFFI",
    "Builder",
    "BuilderConfig",
    "NetworkDefinition",
    "INetworkDefinition",
    "ITensor",
    "Dims",
    "Weights",
    "DataType",
    "ElementWiseOperation",
    "PoolingType",
    "ActivationType",
    "ScaleMode",
    "UnaryOperation",
    "ReduceOperation",
    "MatrixOperation",
    "TopKOperation",
    "MemoryPoolType",
    "OptProfileSelector",
    "BuilderFlag",
    "register_op",
    "get_op_translator",
]
