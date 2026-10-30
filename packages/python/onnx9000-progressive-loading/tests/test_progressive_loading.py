"""Comprehensive unit tests for `onnx9000.progressive_loading` and `onnx9000_progressive_loading`."""

import json
import struct

import pytest
from onnx9000.progressive_loading import ProgressiveLoading
from onnx9000_progressive_loading import ProgressiveLoading as LegacyProgressiveLoading


def test_progressive_loading_init_and_process() -> None:
    """Verify initialization validation and process method across both namespaces."""
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        ProgressiveLoading(chunk_size=0)

    loader = LegacyProgressiveLoading(chunk_size=4)
    assert loader.process("model.safetensors") == "Progressive Loading processed model.safetensors"


def test_safetensors_header_and_byte_range() -> None:
    """Verify Safetensors header parsing and tensor byte range computation."""
    with pytest.raises(ValueError, match="at least 8 bytes"):
        ProgressiveLoading.parse_safetensors_header(b"short")

    header_dict = {
        "weight": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]},
        "bad_entry": {"dtype": "F32"},
    }
    raw_json = json.dumps(header_dict).encode("utf-8")
    prefix = struct.pack("<Q", len(raw_json)) + raw_json

    with pytest.raises(ValueError, match="Buffer truncated"):
        ProgressiveLoading.parse_safetensors_header(prefix[:-2])

    parsed, data_offset = ProgressiveLoading.parse_safetensors_header(prefix + b"\x00" * 16)
    assert data_offset == 8 + len(raw_json)
    assert ProgressiveLoading.compute_tensor_byte_range(parsed, data_offset, "weight") == (
        data_offset,
        data_offset + 16,
    )

    with pytest.raises(KeyError, match="not found"):
        ProgressiveLoading.compute_tensor_byte_range(parsed, data_offset, "missing")

    with pytest.raises(KeyError, match="invalid data_offsets"):
        ProgressiveLoading.compute_tensor_byte_range(parsed, data_offset, "bad_entry")


def test_stream_bytes_and_etag() -> None:
    """Verify chunked byte streaming and ETag normalization/validation."""
    loader = ProgressiveLoading(chunk_size=3)
    chunks = list(loader.stream_bytes(b"abcdefgh"))
    assert chunks == [b"abc", b"def", b"gh"]

    assert ProgressiveLoading.validate_etag('"abc123"', "abc123") is True
    assert ProgressiveLoading.validate_etag("", "") is False
    assert ProgressiveLoading.validate_etag("abc", "xyz") is False
