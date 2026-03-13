"""
Export Sub-Package

Transforms the internal IR Graph into an official ONNX ModelProto binary format,
ensuring compliance with ONNX constraints and constraints.
"""

from onnx9000.export.builder import (
    build_model_proto,
    to_onnx,
    to_string,
    validate_model,
)
from onnx9000.export.proto_utils import (
    to_attribute_proto,
    to_node_proto,
    to_tensor_proto,
    to_value_info_proto,
)

__all__ = [
    "to_onnx",
    "to_string",
    "build_model_proto",
    "validate_model",
    "to_tensor_proto",
    "to_value_info_proto",
    "to_attribute_proto",
    "to_node_proto",
]

from onnx9000.export.bundle import create_model_bundle
from onnx9000.export.chunking import (
    compress_weights_to_int8,
    embed_metadata,
    export_with_external_data,
    generate_chunk_manifest,
)
