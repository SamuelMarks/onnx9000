"""Module providing core logic and structural definitions."""

import enum

from onnx9000.core import onnx_pb2


class DType(enum.Enum):
    """Core data type enumeration bridging Python, NumPy, C++, and ONNX.

    Values correspond to ONNX TensorProto.DataType.
    """

    UNDEFINED = onnx_pb2.TensorProto.UNDEFINED
    FLOAT32 = onnx_pb2.TensorProto.FLOAT
    UINT8 = onnx_pb2.TensorProto.UINT8
    INT8 = onnx_pb2.TensorProto.INT8
    UINT16 = onnx_pb2.TensorProto.UINT16
    INT16 = onnx_pb2.TensorProto.INT16
    INT32 = onnx_pb2.TensorProto.INT32
    INT64 = onnx_pb2.TensorProto.INT64
    STRING = onnx_pb2.TensorProto.STRING
    BOOL = onnx_pb2.TensorProto.BOOL
    FLOAT16 = onnx_pb2.TensorProto.FLOAT16
    FLOAT64 = onnx_pb2.TensorProto.DOUBLE
    UINT32 = onnx_pb2.TensorProto.UINT32
    UINT64 = onnx_pb2.TensorProto.UINT64
    BFLOAT16 = onnx_pb2.TensorProto.BFLOAT16


def to_cpp_type(dtype: DType) -> str:
    """Convert a DType to its corresponding native C++ type string."""
    mapping = {
        DType.FLOAT32: "float",
        DType.FLOAT64: "double",
        DType.INT8: "int8_t",
        DType.INT16: "int16_t",
        DType.INT32: "int32_t",
        DType.INT64: "int64_t",
        DType.UINT8: "uint8_t",
        DType.UINT16: "uint16_t",
        DType.UINT32: "uint32_t",
        DType.UINT64: "uint64_t",
        DType.BOOL: "bool",
        DType.STRING: "uint8_t",
        DType.FLOAT16: "uint16_t",
        DType.BFLOAT16: "uint16_t",
    }
    if dtype not in mapping:
        raise ValueError(f"No C++ type mapped for DType: {dtype}")
    return mapping[dtype]


def to_emscripten_type(dtype: DType) -> str:
    """Convert a DType to its corresponding JS TypedArray string for Emscripten."""
    mapping = {
        DType.FLOAT32: "Float32Array",
        DType.FLOAT64: "Float64Array",
        DType.INT8: "Int8Array",
        DType.INT16: "Int16Array",
        DType.INT32: "Int32Array",
        DType.INT64: "BigInt64Array",
        DType.UINT8: "Uint8Array",
        DType.UINT16: "Uint16Array",
        DType.UINT32: "Uint32Array",
        DType.UINT64: "BigUint64Array",
        DType.BOOL: "Uint8Array",
        DType.FLOAT16: "Uint16Array",
        DType.BFLOAT16: "Uint16Array",
    }
    if dtype not in mapping:
        raise ValueError(f"No Emscripten TypedArray mapped for DType: {dtype}")
    return mapping[dtype]
