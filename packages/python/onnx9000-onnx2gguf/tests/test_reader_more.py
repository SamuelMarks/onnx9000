"""Tests for reader more."""

import io
import struct

import pytest
from onnx9000.onnx2gguf.reader import GGUFReader


def test_reader_unknown_value_type():
    """Provides functional implementation."""
    buf = io.BytesIO()
    buf.write(b"GGUF")
    buf.write(struct.pack("<I", 3))  # version
    buf.write(struct.pack("<Q", 0))  # tensor_count
    buf.write(struct.pack("<Q", 1))  # kv_count

    # kv string
    k = b"test"
    buf.write(struct.pack("<Q", len(k)))
    buf.write(k)
    # write an unknown type (999)
    buf.write(struct.pack("<I", 999))

    buf.seek(0)
    with pytest.raises(ValueError):
        GGUFReader(buf)


def test_reader_read_val_unknown():
    """Provides functional implementation."""
    buf = io.BytesIO()
    buf.write(b"GGUF")
    buf.write(struct.pack("<I", 3))  # version
    buf.write(struct.pack("<Q", 0))  # tensor_count
    buf.write(struct.pack("<Q", 0))  # kv_count
    buf.seek(0)
    reader = GGUFReader(buf)

    with pytest.raises(ValueError, match="Unknown value type"):
        reader._read_val("INVALID_TYPE")


def test_reader_unknown_tensor_type():
    """Provides functional implementation."""
    buf = io.BytesIO()
    buf.write(b"GGUF")
    buf.write(struct.pack("<I", 3))  # version
    buf.write(struct.pack("<Q", 1))  # tensor_count
    buf.write(struct.pack("<Q", 0))  # kv_count

    # tensor name
    tname = b"tensor1"
    buf.write(struct.pack("<Q", len(tname)))
    buf.write(tname)

    # dims
    buf.write(struct.pack("<I", 1))  # 1 dim
    buf.write(struct.pack("<Q", 10))  # dim value

    buf.write(struct.pack("<I", 0))  # GGUFTensorType.F32

    # offset
    buf.write(struct.pack("<Q", 0))

    buf.seek(0)
    reader = GGUFReader(buf)

    # mutate the type to force else branch
    reader.tensors[b"tensor1".decode("utf-8")]["type"] = "INVALID"

    with pytest.raises(ValueError, match="Unknown type"):
        reader.get_tensor("tensor1")
