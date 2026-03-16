from onnx9000.frontend.paddle.parsers import PaddleProtobufParser


def test_paddle_parsers_var_desc_dims_and_skips() -> None:
    name = b"\n\x01X"
    dim_unpacked = b"\x10\x03"
    dim_packed = b"\x12\x01\x04"
    sss_unknown = b"\x1d\x00\x00\x00\x00"
    tensor_msg = b"\x08\x02" + dim_unpacked + dim_packed + sss_unknown
    ss_unknown = b"\x15\x00\x00\x00\x00"
    lod_msg = b"\n" + bytes([len(tensor_msg)]) + tensor_msg + ss_unknown
    type_msg = b"\x08\x07\x12" + bytes([len(lod_msg)]) + lod_msg
    var_type = b"\x12" + bytes([len(type_msg)]) + type_msg
    var_data = name + var_type
    parser = PaddleProtobufParser(var_data)
    var = parser.parse_var_desc(len(var_data))
    assert var.name == "X"
    assert var.dtype == 2
    assert var.shape == [3, 4]
