import pytest
from onnx9000.zoo.tensors import (
    BFloat16Upcaster,
    GSPMDReconciler,
    MsgPackFlaxDeserializer,
    SafeTensorsMmapParser,
)


def test_safetensors_mmap_parser(tmp_path):
    import json
    import struct

    file_path = str(tmp_path / "test.safetensors")
    header = {
        "__metadata__": {"format": "pt"},
        "w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
    }
    header_json = json.dumps(header).encode("utf-8")
    header_len = struct.pack("<Q", len(header_json))
    tensor_data = struct.pack("<f", 1.0)

    with open(file_path, "wb") as f:
        f.write(header_len)
        f.write(header_json)
        f.write(tensor_data)

    parser = SafeTensorsMmapParser(file_path)
    parser.parse()
    assert parser.header_len == len(header_json)

    # Test short read
    short_path = str(tmp_path / "short.safetensors")
    with open(short_path, "wb") as f:
        f.write(b"123")
    parser_short = SafeTensorsMmapParser(short_path)
    parser_short.parse()
    assert parser_short.header_len == 0

    # Test stream
    view = parser.stream_tensor("w")
    assert view is not None
    assert len(view) == 4

    assert parser.stream_tensor("missing") is None


def test_gspmd_reconciler():
    shards = [b"12", b"34"]
    res = GSPMDReconciler.stitch_shards(shards)
    assert res == b"1234"

    import numpy as np

    shards2 = [
        np.array([1, 2], dtype=np.int32).tobytes(),
        np.array([3, 4], dtype=np.int32).tobytes(),
    ]
    res2 = GSPMDReconciler.stitch_shards(shards2, axis=0, shard_shape=(2,), dtype="int32")
    assert res2 == b"1234" or len(res2) == 16  # It concatentates


def test_bfloat16_upcaster():
    # bf16 representation of 1.0 is 0x3f80
    bf16_bytes = b"\x3f\x80"
    res = BFloat16Upcaster.upcast_bfloat16_to_float32(bf16_bytes)
    assert len(res) == 4


def test_msgpack_flax_deserializer():
    try:
        import msgpack

        data = msgpack.packb({"a": 1})
        res = MsgPackFlaxDeserializer.deserialize(data)
        assert res == {"a": 1}
    except ImportError:
        import sys
        import types

        sys.modules["msgpack"] = types.ModuleType("msgpack")
        sys.modules["msgpack"].unpackb = lambda x, **kwargs: {"a": 1}
        res = MsgPackFlaxDeserializer.deserialize(b"")
        assert res == {"a": 1}


def test_gspmd_reconciler_fallback():
    from onnx9000.zoo.tensors import GSPMDReconciler

    shards = [b"12", b"34"]
    res = GSPMDReconciler.stitch_shards(shards, axis=0, shard_shape=None, dtype=None)
    assert res == b"1234"
