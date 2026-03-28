"""TFLite exporter for ONNX models.

This package provides tools for converting ONNX models to TFLite format,
including flatbuffer construction, operator mapping, and quantization.
"""

from .compiler.subgraph import compile_graph_to_tflite
from .exporter import TFLiteExporter
from .flatbuffer.builder import FlatBufferBuilder
from .flatbuffer.schema import (
    Buffer,
    BuiltinOperator,
    BuiltinOptions,
    Metadata,
    Model,
    Operator,
    OperatorCode,
    QuantizationParameters,
    SubGraph,
    Tensor,
    TensorType,
)

__all__ = [
    "FlatBufferBuilder",
    "TensorType",
    "BuiltinOperator",
    "BuiltinOptions",
    "OperatorCode",
    "QuantizationParameters",
    "Tensor",
    "Operator",
    "SubGraph",
    "Buffer",
    "Metadata",
    "Model",
    "TFLiteExporter",
    "compile_graph_to_tflite",
]
