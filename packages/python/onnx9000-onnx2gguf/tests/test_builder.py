"""Module docstring."""

import io
import struct

import pytest
from onnx9000.onnx2gguf.builder import GGUFTensorType, GGUFValueType, GGUFWriter


def test_gguf_writer_basic():
    """Provides functional implementation."""
    f = io.BytesIO()
    writer = GGUFWriter(f)
    writer.add_string("general.name", "test")
    writer.add_uint32("general.alignment", 32)
    writer.add_array("tokens", ["a", "b"], GGUFValueType.STRING)
    writer.add_tensor_info("weight", [2, 2], GGUFTensorType.F32)

    writer.write_header_to_file()

    data = f.getvalue()
    assert data[0:4] == b"GGUF"
    assert struct.unpack("<I", data[4:8])[0] == 3
    assert struct.unpack("<Q", data[8:16])[0] == 1  # tensor count
    assert struct.unpack("<Q", data[16:24])[0] == 3  # KV count

    writer.write_tensor_data(b"\x00" * 16)


def test_gguf_writer_all_types():
    """Provides functional implementation."""
    f = io.BytesIO()
    writer = GGUFWriter(f)
    writer.add_uint8("u8", 1)
    writer.add_int8("i8", -1)
    writer.add_uint16("u16", 1)
    writer.add_int16("i16", -1)
    writer.add_uint32("u32", 1)
    writer.add_int32("i32", -1)
    writer.add_float32("f32", 1.5)
    writer.add_uint64("u64", 1)
    writer.add_int64("i64", -1)
    writer.add_float64("f64", 1.5)
    writer.add_bool("b1", True)
    writer.add_bool("b2", False)

    writer.write_header_to_file()
    assert len(f.getvalue()) > 0


def test_gguf_writer_edge_cases():
    """Provides functional implementation."""
    f = io.BytesIO()
    writer = GGUFWriter(f)
    with pytest.raises(ValueError):
        writer._write_string("a" * (2**31))

    with pytest.raises(ValueError):
        writer._write_val("InvalidType", 1)

    with pytest.raises(ValueError):
        writer.add_tensor_info("weight", [2, 2], "InvalidType")

    writer.add_tensor_info("q4", [32], GGUFTensorType.Q4_0)
    writer.add_tensor_info("q41", [32], GGUFTensorType.Q4_1)
    writer.add_tensor_info("q8", [32], GGUFTensorType.Q8_0)
    writer.add_tensor_info("f16", [2], GGUFTensorType.F16)

    with pytest.raises(AssertionError):
        writer.add_tensor_info("bad_q4", [10], GGUFTensorType.Q4_0)
