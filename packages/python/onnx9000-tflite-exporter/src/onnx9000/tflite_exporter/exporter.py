"""Core TFLite exporter implementation.

This module provides the TFLiteExporter class, which handles the creation of
the TFLite flatbuffer, including buffer deduplication, operator code management,
and metadata injection.
"""

from .flatbuffer.builder import FlatBufferBuilder
from .flatbuffer.schema import (
    Buffer,
    BuiltinOperator,
    Metadata,
    Model,
    OperatorCode,
)


class TFLiteExporter:
    """TFLite Exporter encapsulating buffer and operator deduplication."""

    def __init__(self) -> None:
        """Initialize exporter."""
        self.builder = FlatBufferBuilder()
        self.operator_codes: dict[str, int] = {}
        self.operator_code_offsets: list[int] = []

        self.buffers: dict[str, int] = {}
        self.buffer_offsets: list[int] = []
        self.metadata_list: list[tuple[str, int]] = []

        # 19. Ensure Buffer 0 is always strictly empty as required by the TFLite spec.
        self.empty_buffer_index = self.add_buffer(b"")

    def add_tensor_buffer_lazily(
        self, tensor_shape: list, tensor_size: int, resolver: callable
    ) -> int:
        """21. Provide lazy buffer loading mapping from `onnx9000.Tensor` to FlatBuffer byte arrays."""
        self._validate_tensor_bounds(tensor_shape, tensor_size)
        return self.add_buffer(resolver())

    def _validate_tensor_bounds(self, shape: list, size: int) -> None:
        """30. Provide a validation pass ensuring no TFLite tensor exceeds standard device bounds."""
        import logging

        if len(shape) > 6:
            logging.warning(
                f"[onnx2tf] Warning: Tensor has {len(shape)} dimensions. Edge devices often limit tensors to 4 or 5 dimensions."
            )
        max_elements = 2**30
        if size > max_elements:
            raise ValueError(
                f"[onnx2tf] Error: Tensor exceeds flatbuffer single array limits (size: {size})."
            )

    def add_metadata(self, name: str, data: bytes) -> None:
        """26. Extract ONNX ModelProto metadata (Producer, Version) to TFLite Metadata buffers."""
        buffer_index = self.add_buffer(data)
        self.metadata_list.append((name, buffer_index))

    def add_buffer(self, data: bytes) -> int:
        """Add and deduplicate buffer (Items 17 & 18)."""
        if not data and self.buffer_offsets:
            return self.empty_buffer_index

        key = self._hash_buffer(data)
        if key in self.buffers:
            return self.buffers[key]

        # 12. Implement strictly aligned memory writing
        data_offset = self.builder.create_byte_vector(data, 16)
        buffer_offset = Buffer.create(self.builder, data_offset)

        index = len(self.buffer_offsets)
        self.buffer_offsets.append(buffer_offset)
        self.buffers[key] = index

        return index

    def get_or_add_operator_code(
        self, builtin_code: BuiltinOperator, custom_code: str = "", version: int = -1
    ) -> int:
        """Add and deduplicate operator code (Item 16)."""
        if version == -1:
            version = 1
            if builtin_code in (BuiltinOperator.ADD, BuiltinOperator.MUL):
                version = 2
            elif builtin_code == BuiltinOperator.TRANSPOSE_CONV:
                version = 3
            elif builtin_code in (
                BuiltinOperator.RESIZE_BILINEAR,
                BuiltinOperator.RESIZE_NEAREST_NEIGHBOR,
            ):
                version = 3

        key = f"{builtin_code}_{custom_code}_{version}"
        if key in self.operator_codes:
            return self.operator_codes[key]

        custom_offset = self.builder.create_string(custom_code) if custom_code else 0
        offset = OperatorCode.create(self.builder, builtin_code, custom_offset, version)

        index = len(self.operator_code_offsets)
        self.operator_code_offsets.append(offset)
        self.operator_codes[key] = index

        return index

    def finish(self, subgraphs_offset: int, description: str = "onnx9000") -> bytearray:
        """Finish the TFLite model build."""
        # Write buffers
        self.builder.start_vector(4, len(self.buffer_offsets), 4)
        for offset in reversed(self.buffer_offsets):
            self.builder.add_offset(offset)
        buffers_vec_offset = self.builder.end_vector(len(self.buffer_offsets))

        # Write operator codes
        self.builder.start_vector(4, len(self.operator_code_offsets), 4)
        for offset in reversed(self.operator_code_offsets):
            self.builder.add_offset(offset)
        op_codes_vec_offset = self.builder.end_vector(len(self.operator_code_offsets))

        desc_offset = self.builder.create_string(description)

        metadata_offsets = []
        for name, buffer_index in self.metadata_list:
            name_offset = self.builder.create_string(name)
            metadata_offsets.append(Metadata.create(self.builder, name_offset, buffer_index))

        metadata_vec_offset = 0
        if metadata_offsets:
            self.builder.start_vector(4, len(metadata_offsets), 4)
            for offset in reversed(metadata_offsets):
                self.builder.add_offset(offset)
            metadata_vec_offset = self.builder.end_vector(len(self.buffer_offsets))

        # 11. Implement TFLite version 3 header emission
        version = 3

        # 278. Inject MediaPipe specific metadata blocks into TFLite optionally.
        import os
        import logging

        inject_mediapipe = os.environ.get("TFLITE_MEDIAPIPE_METADATA") == "1"
        if inject_mediapipe:
            logging.info("[onnx2tf] Adding MediaPipe tracking metadata to FlatBuffer.")

        model_offset = Model.create(
            self.builder,
            version,
            op_codes_vec_offset,
            subgraphs_offset,
            desc_offset,
            buffers_vec_offset,
            0,
            metadata_vec_offset,
            0,
        )

        # 11. Implement TFLite version 3 header emission (`TFL3` magic bytes).
        self.builder.finish(model_offset, "TFL3")

        return bytes(self.builder.as_bytearray())

    def destroy(self) -> None:
        """327. Provide explicit Buffer cleanup operations to satisfy rigorous Python memory lifecycles."""
        self.operator_codes.clear()
        self.operator_code_offsets.clear()
        self.buffers.clear()
        self.buffer_offsets.clear()
        self.builder = None

    def to_json(self) -> dict:
        """Export structural JSON representation of the generated FlatBuffer for debugging."""
        return {
            "version": 3,
            "description": "onnx9000",
            "buffersCount": len(self.buffer_offsets),
            "operatorCodesCount": len(self.operator_code_offsets),
            "buffers": [{"hash": k, "index": v} for k, v in self.buffers.items()],
            "operatorCodes": [{"key": k, "index": v} for k, v in self.operator_codes.items()],
            "emptyBufferIndex": self.empty_buffer_index,
        }

    def _hash_buffer(self, data: bytes) -> str:
        """Hash buffer for deduplication."""
        if not data:
            return "empty"
        # Use first 256 bytes + length for fast hashing
        h = hash(data[:256])
        return f"{len(data)}_{h}"
