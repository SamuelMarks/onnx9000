"""Tests the parsers cov final module functionality."""

from onnx9000.converters.paddle.parsers import PaddleGraph, PaddleProtobufParser


def test_paddle_graph_empty_blocks() -> None:
    """Tests the paddle graph empty blocks functionality."""
    g = PaddleGraph()
    assert g.get_main_block().idx == 0
    assert g.topological_sort() == []


def test_paddle_parsers_skip_field() -> None:
    """Tests the paddle parsers skip field functionality."""
    op = b"\n\x0b\n\x04in_1\x12\x01A\x18\x01\x1a\x04relu"
    parser = PaddleProtobufParser(op)
    node = parser.parse_op_desc(len(op))
    assert "in_1" in node.inputs
    assert node.inputs["in_1"] == ["A"]


def test_paddle_parsers_op_desc_extras() -> None:
    """Tests the paddle parsers op desc extras functionality."""
    out_var = b"\n\x05out_1\x12\x01B\x18\x01"
    attr1 = b"\n\x05attr1\x10\x02\x18\x03P\x010\x01\xf8\x06\x01"
    is_target = b"(\x01"
    unknown_op_field = b"\xf8\x06\x01"
    op = (
        b"\x12"
        + bytes([len(out_var)])
        + out_var
        + b'"'
        + bytes([len(attr1)])
        + attr1
        + is_target
        + unknown_op_field
    )
    parser = PaddleProtobufParser(op)
    node = parser.parse_op_desc(len(op))
    assert "out_1" in node.outputs
    assert node.outputs["out_1"] == ["B"]
    assert node.is_target is True


def test_paddle_parsers_var_desc() -> None:
    """Tests the paddle parsers var desc functionality."""
    name = b"\n\x01X"
    tensor_msg = b"\x08\x02"
    lod_msg = b"\n" + bytes([len(tensor_msg)]) + tensor_msg
    type_msg = b"\x08\x07\x12" + bytes([len(lod_msg)]) + lod_msg + b"\x18\x01"
    var_type = b"\x12" + bytes([len(type_msg)]) + type_msg
    persistable = b"\x18\x01"
    unknown = b"\xf8\x06\x01"
    var_data = name + var_type + persistable + unknown
    parser = PaddleProtobufParser(var_data)
    var = parser.parse_var_desc(len(var_data))
    assert var.name == "X"
    assert var.persistable is True
    assert var.dtype == 2


def test_paddle_parsers_block_desc() -> None:
    """Tests the paddle parsers block desc functionality."""
    idx = b"\x08\x01"
    parent_idx = b"\x10\x00"
    var = b"\x1a\x00"
    op = b'"\x00'
    unknown = b"\xf8\x06\x01"
    block_data = idx + parent_idx + var + op + unknown
    parser = PaddleProtobufParser(block_data)
    block = parser.parse_block_desc(len(block_data))
    assert block.idx == 1
    assert block.parent_idx == 0


def test_paddle_parsers_program_desc() -> None:
    """Tests the paddle parsers program desc functionality."""
    block_data = b"\x08\x01\x10\x00"
    block_msg = b"\n" + bytes([len(block_data)]) + block_data
    version_msg = b"\x08\x01\x10\x01"
    version = b"\x12" + bytes([len(version_msg)]) + version_msg
    unknown = b"\xf8\x06\x01"
    prog_data = block_msg + version + unknown
    parser = PaddleProtobufParser(prog_data)
    graph = parser.parse_program_desc(len(prog_data))
    assert len(graph.blocks) == 1
    assert graph.version == 1


def test_paddle_parsers_framework() -> None:
    """Tests the paddle parsers framework functionality."""
    block_data = b"\x08\x01\x10\x00"
    block_msg = b"\n" + bytes([len(block_data)]) + block_data
    parser = PaddleProtobufParser(block_msg)
    graph = parser.parse_program_desc(len(block_msg))
    assert len(graph.blocks) == 1


def test_paddle_parsers_framework_loop() -> None:
    """Tests the paddle parsers framework loop functionality."""
    prog_msg = b"\n\x00"
    framework_data = prog_msg + b"\xf8\x06\x01"
    parser = PaddleProtobufParser(framework_data)
    graph = parser.parse_framework()
    assert len(graph.blocks) == 1
