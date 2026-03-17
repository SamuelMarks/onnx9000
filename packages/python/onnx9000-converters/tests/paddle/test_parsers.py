import logging
from onnx9000.converters.paddle.parsers import (
    PADDLE_TO_ONNX_VERSION,
    PaddleBlock,
    PaddleGraph,
    PaddleNode,
    PaddleProtobufParser,
    PaddleVar,
    fallback_paddle_op,
    get_opset_version,
    load_paddle_model,
    log_unsupported_paddle_node,
    map_paddle_dtype,
)


def test_mappings() -> None:
    assert PADDLE_TO_ONNX_VERSION["2.0.0"] == 11
    assert map_paddle_dtype(5) == 1
    assert map_paddle_dtype(999) == 1


def test_paddlegraph_topological_sort() -> None:
    n1 = PaddleNode("feed", "feed", outputs={"Out": ["n1_out"]})
    n2 = PaddleNode("relu", "relu", inputs={"X": ["n1_out"]}, outputs={"Out": ["n2_out"]})
    n3 = PaddleNode("fetch", "fetch", inputs={"X": ["n2_out"]})
    b = PaddleBlock(0, -1, {}, [n3, n2, n1])
    graph = PaddleGraph([b])
    sorted_nodes = graph.topological_sort()
    assert sorted_nodes[0].name == "feed"
    assert sorted_nodes[1].name == "relu"
    assert sorted_nodes[2].name == "fetch"


def test_paddlegraph_empty() -> None:
    graph = PaddleGraph()
    assert graph.topological_sort() == []
    assert graph.extract_inputs() == []
    assert graph.extract_outputs() == []


def test_paddlegraph_extract_inputs_outputs() -> None:
    v1 = PaddleVar("in1", is_data=True)
    v2 = PaddleVar("weight1", is_data=True, persistable=True)
    v3 = PaddleVar("feed", is_data=True)
    v4 = PaddleVar("out1", is_target=True)
    v5 = PaddleVar("fetch", is_target=True)
    b = PaddleBlock(0, -1, {"in1": v1, "weight1": v2, "feed": v3, "out1": v4, "fetch": v5})
    graph = PaddleGraph([b])
    inputs = graph.extract_inputs()
    assert len(inputs) == 1
    assert inputs[0].name == "in1"
    outputs = graph.extract_outputs()
    assert len(outputs) == 1
    assert outputs[0].name == "out1"


def test_paddle_protobuf_parser_varint() -> None:
    parser = PaddleProtobufParser(b"\x08")
    assert parser.read_varint() == 8


def test_paddle_protobuf_parser_string() -> None:
    parser = PaddleProtobufParser(b"\x04test")
    assert parser.read_string() == "test"


def test_paddle_protobuf_parser_op_desc() -> None:
    data = b'\x1a\x04relu\n\x07\n\x01X\x12\x02in\x12\n\n\x03Out\x12\x03out"\t\n\x03val\x10\x03\x18*'
    parser = PaddleProtobufParser(data)
    node = parser.parse_op_desc(len(data))
    assert node.op_type == "relu"
    assert node.inputs["X"] == ["in"]
    assert node.outputs["Out"] == ["out"]
    assert node.attrs["val"] == 42


def test_paddle_protobuf_parser_var_desc() -> None:
    data = b"\n\x02v1\x18\x01"
    parser = PaddleProtobufParser(data)
    var = parser.parse_var_desc(len(data))
    assert var.name == "v1"
    assert var.persistable is True


def test_paddle_protobuf_parser_var_desc_deep() -> None:
    shape_data = b"\x01\x02\x03"
    shape_msg = b"\x12" + bytes([len(shape_data)]) + shape_data
    dtype_msg = b"\x08\x05"
    tensor_data = dtype_msg + shape_msg
    tensor_msg = b"\n" + bytes([len(tensor_data)]) + tensor_data
    lod_msg = b"\x12" + bytes([len(tensor_msg)]) + tensor_msg
    type_info_msg = b"\x12" + bytes([len(lod_msg)]) + lod_msg
    name_msg = b"\n\x02v1"
    data = name_msg + type_info_msg
    parser = PaddleProtobufParser(data)
    var = parser.parse_var_desc(len(data))
    assert var.name == "v1"
    assert var.dtype == 5
    assert var.shape == [1, 2, 3]


def test_paddle_protobuf_parser_parse_block_and_program() -> None:
    block_data = b'\x08\x00\x10\x00\x1a\x04\n\x02v1"\x06\x1a\x04relu'
    program_data = b"\n" + bytes([len(block_data)]) + block_data + b"\x12\x02\x08\x0b"
    parser = PaddleProtobufParser(program_data)
    graph = parser.parse_program_desc(len(program_data))
    assert graph.version == 11
    assert len(graph.blocks) == 1
    assert graph.blocks[0].idx == 0
    assert "v1" in graph.blocks[0].vars
    assert len(graph.blocks[0].ops) == 1
    assert graph.blocks[0].ops[0].op_type == "relu"
    parser = PaddleProtobufParser(program_data)
    graph2 = parser.parse_framework()
    assert graph2.version == 11
    assert len(graph2.blocks) == 1


def test_paddle_protobuf_parser_attr_types() -> None:
    import struct

    f_bytes = struct.pack("<f", 1.0)
    data = (
        b'\x1a\x02op"'
        + bytes([len(b"\n\x02f1\x10\x04%" + f_bytes)])
        + b"\n\x02f1\x10\x04%"
        + f_bytes
        + b'"\t\n\x02s1\x10\x05*\x01s"\x08\n\x02b1\x10\nP\x01'
    )
    parser = PaddleProtobufParser(data)
    node = parser.parse_op_desc(len(data))
    assert node.attrs["f1"] == 1.0
    assert node.attrs["s1"] == "s"
    assert node.attrs["b1"] is True


def test_paddle_load_model_feed_fetch() -> None:
    op1_sub = b"\n\x03Out\x12\x03in1"
    op1 = b"\x1a\x04feed\x12" + bytes([len(op1_sub)]) + op1_sub
    op2_sub = b"\n\x01X\x12\x04out1"
    op2 = b"\x1a\x05fetch\n" + bytes([len(op2_sub)]) + op2_sub
    var1 = b"\n\x03in1"
    var2 = b"\n\x04out1"
    block_data = (
        b"\x08\x00\x10\x00"
        + b"\x1a"
        + bytes([len(var1)])
        + var1
        + b"\x1a"
        + bytes([len(var2)])
        + var2
        + b'"'
        + bytes([len(op1)])
        + op1
        + b'"'
        + bytes([len(op2)])
        + op2
    )
    program_data = b"\n" + bytes([len(block_data)]) + block_data
    graph = load_paddle_model(program_data)
    assert graph.blocks[0].vars["in1"].is_data is True
    assert graph.blocks[0].vars["out1"].is_target is True


def test_load_paddle_model() -> None:
    graph = load_paddle_model(b"", b"mock_params")
    assert graph.tensors["mock"] == b"\x00"


def test_get_opset_version() -> None:
    assert get_opset_version(100) == 15


def test_fallback_paddle_op() -> None:
    n = PaddleNode("n1", "some_op")
    fallback_paddle_op(n)
    assert n.op_type == "Custom_Paddle_some_op"


def test_log_unsupported_paddle_node(caplog) -> None:
    n = PaddleNode("n1", "some_op")
    with caplog.at_level(logging.WARNING):
        log_unsupported_paddle_node(n)
    assert "Unsupported Paddle op: some_op" in caplog.text
