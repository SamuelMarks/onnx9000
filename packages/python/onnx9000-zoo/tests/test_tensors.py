import json
import os
import struct

import pytest
from onnx9000.zoo.tensors import (
    BFloat16Upcaster,
    GSPMDReconciler,
    MsgPackFlaxDeserializer,
    SafeTensorsMmapParser,
)


def test_safe_tensors_mmap_parser(tmpdir):
    """Test safetensors parser."""
    file_path = os.path.join(tmpdir, "model.safetensors")

    header = {
        "__metadata__": {"format": "pt"},
        "tensor1": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]},
    }

    header_bytes = json.dumps(header).encode("utf-8")
    header_len = len(header_bytes)

    # 8 bytes length, then header, then data
    with open(file_path, "wb") as f:
        f.write(struct.pack("<Q", header_len))
        f.write(header_bytes)
        f.write(b"\x01" * 16)

    parser = SafeTensorsMmapParser(file_path)
    parser.parse()

    assert parser.header_len == header_len
    assert "tensor1" in parser.header

    tensor_data = parser.stream_tensor("tensor1")
    assert tensor_data is not None
    assert len(tensor_data) == 16
    assert tensor_data[0] == 1

    missing_data = parser.stream_tensor("tensor2")
    assert missing_data is None


def test_safe_tensors_invalid(tmpdir):
    file_path = os.path.join(tmpdir, "invalid.safetensors")
    with open(file_path, "wb") as f:
        f.write(b"\x00\x00")
    parser = SafeTensorsMmapParser(file_path)
    parser.parse()
    assert parser.header_len == 0


def test_gspmd_reconciler():
    """Test GSPMD reconciler."""
    shards = [b"\x01\x02", b"\x03\x04"]
    stitched = GSPMDReconciler.stitch_shards(shards)
    assert stitched == b"\x01\x02\x03\x04"


def test_gspmd_reconciler_axis():
    """Test GSPMD reconciler with numpy arrays."""
    import numpy as np

    arr1 = np.array([[1, 2]], dtype=np.uint8)
    arr2 = np.array([[3, 4]], dtype=np.uint8)
    shards = [arr1.tobytes(), arr2.tobytes()]
    stitched = GSPMDReconciler.stitch_shards(shards, axis=1, shard_shape=(1, 2), dtype="uint8")
    expected = np.array([[1, 2, 3, 4]], dtype=np.uint8).tobytes()
    assert stitched == expected


def test_bfloat16_upcaster():
    """Test BF16 upcaster."""
    # A fake bf16 sequence
    bf16 = b"\x12\x34\x56\x78"
    fp32 = BFloat16Upcaster.upcast_bfloat16_to_float32(bf16)

    assert len(fp32) == 8
    # Should pad 00 00 to the left (little endian)
    # the code wrote result[i*2]=0, result[i*2+1]=0, result[i*2+2]=bf16[i], result[i*2+3]=bf16[i+1]
    assert fp32[0] == 0
    assert fp32[1] == 0
    assert fp32[2] == 0x12
    assert fp32[3] == 0x34

    assert fp32[4] == 0
    assert fp32[5] == 0
    assert fp32[6] == 0x56
    assert fp32[7] == 0x78


def test_msgpack_flax_deserializer():
    """Test flax msgpack deserializer."""
    msgpack = pytest.importorskip("msgpack")

    data = msgpack.packb({"key": "value"})
    res = MsgPackFlaxDeserializer.deserialize(data)
    assert res["key"] == b"value" or res["key"] == "value"


def test_msgpack_flax_deserializer_no_msgpack():
    """Test without msgpack."""
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "msgpack":
            raise ImportError("No module named 'msgpack'")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = mock_import
    try:
        with pytest.raises(ImportError, match="No module named 'msgpack'"):
            MsgPackFlaxDeserializer.deserialize(b"something")
    finally:
        builtins.__import__ = real_import
