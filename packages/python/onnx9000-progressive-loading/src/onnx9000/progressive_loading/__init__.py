"""Progressive and streaming Safetensors/ONNX model loader for ONNX9000.

Provides HTTP byte-range header parsing, chunked streaming serialization,
tensor slice byte-range calculation, and ETag validation for resumable model loading.
"""

import json
import struct
from collections.abc import Iterator
from typing import TypeAlias

from onnx9000.core.exceptions import SafetensorsError

HeaderEntry: TypeAlias = dict[str, str | list[int]]
HeaderMap: TypeAlias = dict[str, HeaderEntry | dict[str, str]]


class ProgressiveLoading:
    """Progressive streaming loader for Safetensors and ONNX weight files."""

    def __init__(self, chunk_size: int = 65536) -> None:
        """Initialize the progressive loader with a default byte chunk size.

        Args:
            chunk_size: Byte chunk size for range requests and streaming (must be positive).

        Raises:
            ValueError: If `chunk_size <= 0`.
        """
        if chunk_size <= 0:
            raise SafetensorsError(f"chunk_size must be positive, got {chunk_size}")
        self.chunk_size: int = chunk_size

    def process(self, arg: str) -> str:
        """Process a model path or URI for progressive loading.

        Args:
            arg: Model path or URI identifier.

        Returns:
            Status summary string.
        """
        return f"Progressive Loading processed {arg}"

    @staticmethod
    def parse_safetensors_header(buffer: bytes) -> tuple[HeaderMap, int]:
        """Parse the 8-byte little-endian header length and JSON metadata from a Safetensors prefix.

        Args:
            buffer: Raw byte prefix containing at least the 8-byte length and JSON header.

        Returns:
            Tuple of `(parsed_header_dict, data_start_offset)`.

        Raises:
            ValueError: If `buffer` is shorter than 8 bytes or shorter than `8 + header_len`.
        """
        if len(buffer) < 8:
            raise SafetensorsError("Safetensors buffer must be at least 8 bytes long")
        (header_len,) = struct.unpack("<Q", buffer[:8])
        data_offset = 8 + header_len
        if len(buffer) < data_offset:
            raise SafetensorsError(
                f"Buffer truncated: expected {data_offset} bytes for header, got {len(buffer)}"
            )
        header_json = buffer[8:data_offset].decode("utf-8")
        parsed: HeaderMap = json.loads(header_json)
        return parsed, data_offset

    @staticmethod
    def compute_tensor_byte_range(
        header: HeaderMap,
        data_offset: int,
        tensor_name: str,
    ) -> tuple[int, int]:
        """Compute the absolute `(start_byte, end_byte)` HTTP Range for a named tensor.

        Args:
            header: Parsed Safetensors header dictionary.
            data_offset: Byte offset where the raw data section begins (`8 + header_len`).
            tensor_name: Name of the tensor to locate.

        Returns:
            Tuple of `(absolute_start_byte, absolute_end_byte)`.

        Raises:
            KeyError: If `tensor_name` is not found in `header` or lacks valid `data_offsets`.
        """
        entry = header.get(tensor_name)
        if not isinstance(entry, dict):
            raise KeyError(f"Tensor '{tensor_name}' not found in Safetensors header")
        offsets = entry.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not isinstance(offsets[0], int)
            or not isinstance(offsets[1], int)
        ):
            raise KeyError(f"Tensor '{tensor_name}' has invalid data_offsets")
        return data_offset + offsets[0], data_offset + offsets[1]

    def stream_bytes(self, payload: bytes) -> Iterator[bytes]:
        """Yield slices of `payload` in increments of `self.chunk_size`.

        Args:
            payload: Raw binary buffer to stream.

        Yields:
            Byte chunks of length up to `self.chunk_size`.
        """
        for offset in range(0, len(payload), self.chunk_size):
            yield payload[offset : offset + self.chunk_size]

    @staticmethod
    def validate_etag(actual_etag: str, expected_etag: str) -> bool:
        """Verify that an HTTP ETag matches the expected ETag before resuming a range download.

        Args:
            actual_etag: ETag header string returned by the server.
            expected_etag: Expected ETag string recorded when the download started.

        Returns:
            `True` if normalized ETags match and are non-empty, `False` otherwise.
        """
        norm_actual = actual_etag.strip().strip('"')
        norm_expected = expected_etag.strip().strip('"')
        return bool(norm_actual) and norm_actual == norm_expected


__all__ = ["ProgressiveLoading"]
