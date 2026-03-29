"""Mapping utilities for ONNX to TFLite types and shapes.

This module provides functions to convert ONNX data types and tensor shapes
to their corresponding TFLite representations.
"""

import logging
from typing import Optional, Union

from onnx9000.core.ir import DType

from ..flatbuffer.schema import TensorType

logger = logging.getLogger(__name__)


def map_onnx_type_to_tflite(dtype: str, name: Optional[str] = None) -> TensorType:
    """Map ONNX dtypes to TFLite dtypes."""
    # 55. Map ONNX FLOAT -> TFLite FLOAT32.
    if dtype == "float32":
        return TensorType.FLOAT32
    # 56. Map ONNX FLOAT16 -> TFLite FLOAT16.
    elif dtype == "float16":
        return TensorType.FLOAT16
    # 57. Map ONNX INT32 -> TFLite INT32.
    elif dtype == "int32":
        return TensorType.INT32
    # 58. Map ONNX INT64 -> TFLite INT64.
    # 314. Prevent Int64 tensor generation inside mobile targets (converting natively to Int32 and warning user).
    elif dtype == "int64":
        if name:
            logger.warning(
                f"[onnx2tf] Warning: Downcasting Int64 tensor '{name}' to Int32 for mobile compatibility."
            )
        return TensorType.INT32
    # 59. Map ONNX INT8 -> TFLite INT8.
    elif dtype == "int8":
        return TensorType.INT8
    # 60. Map ONNX UINT8 -> TFLite UINT8.
    elif dtype == "uint8":
        return TensorType.UINT8
    # 61. Map ONNX BOOL -> TFLite BOOL.
    elif dtype == "bool":
        return TensorType.BOOL
    # 62. Map ONNX STRING -> TFLite STRING.
    elif dtype == "string":
        return TensorType.STRING
    # 63. Handle ONNX DOUBLE (Float64) gracefully (downcast to Float32).
    elif dtype == "float64":
        return TensorType.FLOAT32
    else:
        # 73. Provide fallback casting if TFLite lacks an op signature for a specific type.
        return TensorType.FLOAT32


def map_onnx_shape_to_tflite(shape: list) -> list[int]:
    """Map ONNX shapes to TFLite shapes."""
    # 64. Map empty ONNX shapes [] to TFLite scalar shapes [].
    if not shape:
        return []

    mapped = []
    for dim in shape:
        if isinstance(dim, int):
            # 65. Map dynamic ONNX shapes [-1, 224, 224, 3] safely.
            mapped.append(dim if dim >= 0 else -1)
        else:
            # Dynamic string dimensions become -1 in TFLite
            mapped.append(-1)
    return mapped


def create_shape_signature(shape: list) -> list[int]:
    """66. Emit ShapeSignature vectors for TFLite dynamic shapes."""
    return map_onnx_shape_to_tflite(shape)
