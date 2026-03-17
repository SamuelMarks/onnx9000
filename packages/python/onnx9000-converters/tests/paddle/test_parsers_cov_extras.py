"""Tests the parsers cov extras module functionality."""

import pytest
from onnx9000.converters.paddle.parsers import PaddleProtobufParser


def test_paddle_parsers_var_desc_dims_and_skips() -> None:
    """Tests the paddle parsers var desc dims and skips functionality."""
    name = b"\n\x01X"
    skip_wire1 = b"\x19\x00\x00\x00\x00\x00\x00\x00\x00"
    skip_wire2 = b'"\x04abcd'
    skip_wire5 = b"-\x00\x00\x00\x00"
    dim_unpacked = b"\x10\x03"
    dim_packed = b"\x12\x01\x04"
    tensor_msg = b"\x08\x02" + dim_unpacked + dim_packed + skip_wire1 + skip_wire2 + skip_wire5
    lod_msg = b"\n" + bytes([len(tensor_msg)]) + tensor_msg
    type_msg = b"\x08\x07\x12" + bytes([len(lod_msg)]) + lod_msg
    var_type = b"\x12" + bytes([len(type_msg)]) + type_msg
    var_data = name + var_type
    parser = PaddleProtobufParser(var_data)
    var = parser.parse_var_desc(len(var_data))
    assert var.name == "X"
    assert var.dtype == 2


def test_paddle_parsers_skip_field_error() -> None:
    """Tests the paddle parsers skip field error functionality."""
    bad_wire = b"\n\x01X\xfb\x06\x01"
    parser = PaddleProtobufParser(bad_wire)
    with pytest.raises(ValueError, match="Unknown wire type: 3"):
        parser.parse_var_desc(len(bad_wire))
