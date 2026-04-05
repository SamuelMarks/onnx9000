"""Tests for reader."""

from onnx9000.tflite_exporter.exporter import TFLiteExporter
from onnx9000.tflite_exporter.flatbuffer.builder import FlatBufferBuilder
from onnx9000.tflite_exporter.flatbuffer.reader import FlatBufferReader
from onnx9000.tflite_exporter.flatbuffer.schema import BuiltinOperator


def test_flatbuffer_reader():
    """Provides functional implementation."""
    builder = FlatBufferBuilder()

    str_offset = builder.create_string("test_string")

    builder.start_object(2)
    builder.add_field_offset(0, str_offset, 0)
    builder.add_field_int32(1, 42, 0)
    root = builder.end_object()

    builder.finish(root, "TFL3")
    buf = builder.as_bytearray()

    reader = FlatBufferReader(buf)
    assert reader.check_magic_bytes("TFL3")

    table_loc = reader.get_root()

    str_val = reader.get_string(table_loc, 0)
    assert str_val == "test_string"

    int_val = reader.get_int32(table_loc, 1, 0)
    assert int_val == 42


def test_tflite_exporter_reader():
    """Provides functional implementation."""
    exporter = TFLiteExporter()
    exporter.add_metadata("TestMeta", b"\x01\x02\x03")
    exporter.get_or_add_operator_code(BuiltinOperator.ADD)

    buf = exporter.finish(0, "test_desc")

    reader = FlatBufferReader(buf)
    assert reader.check_magic_bytes("TFL3")

    table_loc = reader.get_root()

    desc = reader.get_string(table_loc, 3)
    assert desc == "test_desc"

    version = reader.get_int32(table_loc, 0, 0)
    assert version == 3


def test_reader_edge_cases():
    """Provides functional implementation."""
    reader = FlatBufferReader(b"short")
    assert not reader.check_magic_bytes("TFL3")

    builder = FlatBufferBuilder()
    builder.start_object(1)
    root = builder.end_object()
    builder.finish(root, "TFL3")

    reader2 = FlatBufferReader(builder.as_bytearray())
    table_loc = reader2.get_root()
    assert reader2.get_indirect_offset(table_loc, 0) == 0
