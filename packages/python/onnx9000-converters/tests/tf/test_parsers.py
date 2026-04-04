"""Tests the parsers module functionality."""

import logging

from onnx9000.converters.tf.parsers import (
    TF_DTYPE_TO_ONNX,
    TF_TO_ONNX_VERSION_MAPPING,
    ProtobufParser,
    TFGraph,
    TFNode,
    extract_variables,
    fallback_to_custom_op,
    load_h5_model,
    load_keras_v3,
    log_unsupported_node,
    map_tf_shape_to_onnx,
    parse_graphdef,
    parse_saved_model,
    parse_tflite,
)


def test_mappings() -> None:
    """Tests the mappings functionality."""
    assert TF_TO_ONNX_VERSION_MAPPING["1.15.0"] == 11
    assert TF_DTYPE_TO_ONNX[1] == 1


def test_tfgraph_topological_sort() -> None:
    """Tests the tfgraph topological sort functionality."""
    n1 = TFNode("n1", "Placeholder")
    n2 = TFNode("n2", "Relu", inputs=["n1:0"])
    n3 = TFNode("n3", "MatMul", inputs=["n2:0", "n1:0"])
    graph = TFGraph([n3, n2, n1])
    sorted_nodes = graph.topological_sort()
    names = [n.name for n in sorted_nodes]
    assert names.index("n1") < names.index("n2")
    assert names.index("n2") < names.index("n3")


def test_tfgraph_resolve_duplicate_names() -> None:
    """Tests the tfgraph resolve duplicate names functionality."""
    graph = TFGraph([TFNode("n1", "Placeholder"), TFNode("n1", "Relu"), TFNode("n1", "MatMul")])
    graph.resolve_duplicate_names()
    assert [n.name for n in graph.nodes] == ["n1", "n1_1", "n1_2"]


def test_tfgraph_extract_inputs_outputs() -> None:
    """Tests the tfgraph extract inputs outputs functionality."""
    n1 = TFNode("in1", "Placeholder")
    n2 = TFNode("mid", "Relu", inputs=["in1:0"])
    n3 = TFNode("out", "MatMul", inputs=["mid:0"])
    graph = TFGraph([n1, n2, n3])
    inputs = graph.extract_inputs()
    assert [n.name for n in inputs] == ["in1"]
    outputs = graph.extract_outputs()
    assert [n.name for n in outputs] == ["out"]


def test_tfgraph_extract_subgraph() -> None:
    """Tests the tfgraph extract subgraph functionality."""
    n1 = TFNode("n1", "Placeholder")
    n2 = TFNode("n2", "Relu", inputs=["n1:0"])
    n3 = TFNode("n3", "MatMul", inputs=["n2:0"])
    n4 = TFNode("n4", "Unrelated")
    graph = TFGraph([n1, n2, n3, n4])
    sub = graph.extract_subgraph(["n3"])
    assert {n.name for n in sub.nodes} == {"n1", "n2", "n3"}


def test_protobuf_parser_varint() -> None:
    """Tests the protobuf parser varint functionality."""
    parser = ProtobufParser(b"\x08")
    assert parser.read_varint() == 8
    parser = ProtobufParser(b"\x96\x01")
    assert parser.read_varint() == 150


def test_protobuf_parser_bytes_string() -> None:
    """Tests the protobuf parser bytes string functionality."""
    parser = ProtobufParser(b"\x04test")
    assert parser.read_bytes() == b"test"
    parser = ProtobufParser(b"\x04test")
    assert parser.read_string() == "test"


def test_protobuf_parser_node_def() -> None:
    """Tests the protobuf parser node def functionality."""
    data = b'\n\x02n1\x12\x03op1\x1a\x03in1"\x02ab'
    parser = ProtobufParser(data)
    node = parser.parse_node_def(len(data))
    assert node.name == "n1"
    assert node.op == "op1"
    assert node.inputs == ["in1"]


def test_protobuf_parser_graph_def() -> None:
    """Tests the protobuf parser graph def functionality."""
    node_data = b"\n\x02n1\x12\x03op1\x1a\x03in1"
    data = b"\n\x0c" + node_data + b'"\x02ab'
    parser = ProtobufParser(data)
    graph = parser.parse_graph_def()
    assert len(graph.nodes) == 1
    assert graph.nodes[0].name == "n1"
    assert graph.nodes[0].op == "op1"


def test_protobuf_parser_skip_fields() -> None:
    """Tests the protobuf parser skip fields functionality."""
    data = b"\x00\x08"
    data += b"\t\x01\x02\x03\x04\x05\x06\x07\x08"
    data += b"\x12\x02ab"
    data += b"\x1d\x01\x02\x03\x04"
    parser = ProtobufParser(data)
    tag = parser.read_varint()
    parser.skip_field(tag & 7)
    tag = parser.read_varint()
    parser.skip_field(tag & 7)
    tag = parser.read_varint()
    parser.skip_field(tag & 7)
    tag = parser.read_varint()
    parser.skip_field(tag & 7)
    assert parser.offset == len(data)


def test_protobuf_incomplete_varint() -> None:
    """Tests the protobuf incomplete varint functionality."""
    parser = ProtobufParser(b"\x80")
    assert parser.read_varint() == 0


def test_protobuf_parser_node_def_unknown_field() -> None:
    """Tests the protobuf parser node def unknown field functionality."""
    data = b"*\x03unk"
    parser = ProtobufParser(data)
    node = parser.parse_node_def(len(data))
    assert node.name == ""
    assert node.op == ""


def test_protobuf_parser_graph_def_unknown_field() -> None:
    """Tests the protobuf parser graph def unknown field functionality."""
    data = b"*\x03unk"
    parser = ProtobufParser(data)
    graph = parser.parse_graph_def()
    assert len(graph.nodes) == 0


def test_parse_functions() -> None:
    """Tests the parse functions functionality."""
    assert len(parse_graphdef(b"").nodes) == 0
    assert len(parse_saved_model(b"").nodes) == 0
    assert extract_variables("v") == {"v": b"0000"}
    assert load_h5_model(b"").nodes[0].name == "h5_input"
    assert load_keras_v3(b"").nodes[0].name == "keras3_input"
    assert parse_tflite(b"").nodes[0].name == "tflite_input"
    assert map_tf_shape_to_onnx([1, 2, -1, 0]) == [1, 2, -1, -1]
    node = TFNode("n", "op")
    fallback_to_custom_op(node)
    assert node.op == "Custom_op"


def test_h5_parser_stub() -> None:
    """Docstring for D103."""
    from onnx9000.converters.tf.parsers import H5Parser

    parser = H5Parser(b"")
    graph = parser.parse()
    assert graph.nodes[0].name == "h5_input"


def test_load_keras_v3_fallback() -> None:
    """Docstring for D103."""
    import sys
    from unittest.mock import MagicMock, patch

    keras_mock = MagicMock()

    class DummyModel:
        pass

    keras_mock.Model = DummyModel
    sys.modules["keras"] = keras_mock

    with patch("onnx9000.converters.tf.keras_v3_parser.Keras3Parser") as mock_parser:
        mock_graph = MagicMock()
        mock_graph.nodes = [MagicMock()]
        mock_graph.nodes[0].name = "keras3_input"
        mock_parser.return_value.parse.return_value = mock_graph

        # Test with keras.Model
        model = DummyModel()
        graph = load_keras_v3(model)
        assert graph.nodes[0].name == "keras3_input"

        # Test with generic data (not bytes, not keras.Model)
        graph2 = load_keras_v3("string_data")
        assert graph2.nodes[0].name == "keras3_input"


def test_log_unsupported_node(caplog) -> None:
    """Tests the log unsupported node functionality."""
    node = TFNode("n", "op")
    with caplog.at_level(logging.WARNING):
        log_unsupported_node(node)
    assert "Unsupported TF Node encountered: op (name: n)" in caplog.text
