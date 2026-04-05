"""Tests for reader."""

import io

import pytest
from onnx9000.onnx2gguf.builder import GGUFTensorType, GGUFValueType, GGUFWriter
from onnx9000.onnx2gguf.reader import GGUFReader


def test_reader():
    """Provides functional implementation."""
    f = io.BytesIO()
    writer = GGUFWriter(f)
    writer.add_string("general.name", "test")
    writer.add_uint32("general.alignment", 32)
    writer.add_array("tokens", ["a", "b"], GGUFValueType.STRING)
    writer.add_tensor_info("weight", [2, 2], GGUFTensorType.F32)
    writer.add_tensor_info("q4", [32], GGUFTensorType.Q4_0)
    writer.add_tensor_info("q41", [32], GGUFTensorType.Q4_1)
    writer.add_tensor_info("q8", [32], GGUFTensorType.Q8_0)
    writer.add_tensor_info("f16", [2], GGUFTensorType.F16)

    writer.write_header_to_file()
    writer.write_tensor_data(b"\x00" * 16)  # weight
    writer.write_tensor_data(b"\x00" * 18)  # q4
    writer.write_tensor_data(b"\x00" * 20)  # q41
    writer.write_tensor_data(b"\x00" * 34)  # q8
    writer.write_tensor_data(b"\x00" * 4)  # f16

    f.seek(0)
    reader = GGUFReader(f)
    assert reader.kvs["general.name"] == "test"
    assert reader.kvs["tokens"] == ["a", "b"]

    data = reader.get_tensor("weight")
    assert len(data) == 16
    print("q41 offset:", reader.tensors["q41"]["offset"])
    assert len(reader.get_tensor("q4")) == 18
    assert len(reader.get_tensor("q41")) == 20
    assert len(reader.get_tensor("q8")) == 34
    assert len(reader.get_tensor("f16")) == 4

    with pytest.raises(KeyError):
        reader.get_tensor("missing")


def test_reader_invalid():
    """Provides functional implementation."""
    f = io.BytesIO(b"GGUX")
    with pytest.raises(ValueError):
        GGUFReader(f)

    f = io.BytesIO(b"GGUF\x04\x00\x00\x00")
    with pytest.raises(ValueError):
        GGUFReader(f)


def test_reader_all_types():
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
    f.seek(0)
    reader = GGUFReader(f)
    assert reader.kvs["u8"] == 1
    assert reader.kvs["i8"] == -1
    assert reader.kvs["f32"] == 1.5
