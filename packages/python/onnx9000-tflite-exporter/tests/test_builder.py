"""Tests for the FlatBuffer builder used in the TFLite exporter."""

from onnx9000.tflite_exporter.flatbuffer.builder import FlatBufferBuilder


def test_flatbuffer_builder():
    """Test the basic functionality of the FlatBufferBuilder."""
    builder = FlatBufferBuilder()

    str_offset = builder.create_string("test")

    builder.start_object(2)
    builder.add_field_offset(0, str_offset, 0)
    builder.add_field_int32(1, 42, 0)
    root = builder.end_object()

    builder.finish(root, "TEST")
    buf = builder.as_bytearray()

    assert len(buf) > 0
    assert len(buf) % 4 == 0

    # Check magic bytes
    magic = buf[4:8].decode("ascii")
    assert magic == "TEST"


def test_builder_edge_cases():
    """Test edge cases and error conditions in the FlatBufferBuilder."""
    builder = FlatBufferBuilder()

    # Test clear
    builder.clear()
    assert builder.space == 1024
    assert builder.vtable is None

    # Test create_string with bytes
    builder.create_string(b"bytes_string")

    # Test add_field_float64
    builder.start_object(1)
    builder.add_field_float64(0, 3.1415926535, 0.0)
    builder.end_object()

    # Test end_object without start_object
    try:
        builder.end_object()
    except ValueError:
        return None

    # Test grow edge case (simulating initial 0 length or empty bb)
    empty_builder = FlatBufferBuilder()
    empty_builder.bb = bytearray()  # Reset to 0
    empty_builder.grow_buffer()
    assert len(empty_builder.bb) == 1024
