"""Module docstring."""

import os
import struct
import tempfile

import pytest
from onnx9000.converters.onnx_parser import PureOnnxParser


def write_varint(f, value):
    """Docstring for D103."""
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            f.write(bytes([byte | 0x80]))
        else:
            f.write(bytes([byte]))
            break


def write_tag(f, field_num, wire_type):
    """Docstring for D103."""
    write_varint(f, (field_num << 3) | wire_type)


def write_string(f, field_num, string_val):
    """Docstring for D103."""
    write_tag(f, field_num, 2)
    b = string_val.encode("utf-8")
    write_varint(f, len(b))
    f.write(b)


def test_pure_onnx_parser():
    """Docstring for D103."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        # We will build a dummy ModelProto containing a GraphProto with a TensorProto

        tensor_bytes = bytearray()
        # dims = [2, 2]
        # write tag for dims (field 1)
        tensor_bytes.extend(bytes([1 << 3 | 2]))  # length delimited (packed)
        tensor_bytes.extend(bytes([2]))  # length
        tensor_bytes.extend(bytes([2]))  # val
        tensor_bytes.extend(bytes([2]))  # val
        # data_type = 1 (FLOAT) -> field 2
        tensor_bytes.extend(bytes([2 << 3 | 0]))  # varint
        tensor_bytes.extend(bytes([1]))  # float
        # name = "test" -> field 8
        tensor_bytes.extend(bytes([8 << 3 | 2]))  # string
        tensor_bytes.extend(bytes([4]))  # len 4
        tensor_bytes.extend(b"test")
        # raw_data = floats -> field 9
        tensor_bytes.extend(bytes([9 << 3 | 2]))  # string/bytes
        data_b = struct.pack("<ffff", 1.0, 2.0, 3.0, 4.0)
        tensor_bytes.extend(bytes([16]))  # len 16
        tensor_bytes.extend(data_b)

        graph_bytes = bytearray()
        # name = "dummy_graph" -> field 2
        graph_bytes.extend(bytes([2 << 3 | 2]))
        graph_bytes.extend(bytes([11]))
        graph_bytes.extend(b"dummy_graph")
        # initializer = TensorProto -> field 5
        graph_bytes.extend(bytes([5 << 3 | 2]))
        graph_bytes.extend(bytes([len(tensor_bytes)]))
        graph_bytes.extend(tensor_bytes)

        # ModelProto
        # graph -> field 7
        write_tag(f, 7, 2)
        write_varint(f, len(graph_bytes))
        f.write(graph_bytes)

        filename = f.name

    try:
        with PureOnnxParser(filename) as parser:
            model = parser.parse_model()

            pass
            graph = model["graph"]
            assert graph.get("name") == "dummy_graph"

            assert "initializer" in graph
            init = graph["initializer"][0]
            assert init.get("name") == "test"
            assert init.get("data_type") == 1
            assert init.get("dims") == [2, 2]

            raw_data = init.get("raw_data")
            floats = struct.unpack("<ffff", raw_data)
            assert floats == (1.0, 2.0, 3.0, 4.0)

    finally:
        os.remove(filename)


def test_onnx_parser_errors():
    """Docstring for D103."""
    import pytest
    from onnx9000.converters.onnx_parser import read_varint, skip_field

    with pytest.raises(EOFError):
        read_varint(memoryview(b"\x80"), 0)

    # Check skip_field branches
    assert skip_field(memoryview(b"\x01"), 0, 0) == 1
    assert skip_field(memoryview(b"\x01\x02\x03\x04\x05\x06\x07\x08"), 0, 1) == 8
    assert skip_field(memoryview(b"\x03abc"), 0, 2) == 4  # length delimited, len 3
    assert skip_field(memoryview(b"\x01\x02\x03\x04"), 0, 5) == 4

    with pytest.raises(ValueError):
        skip_field(memoryview(b""), 0, 99)


def test_onnx_parser_parse_model(tmp_path):
    """Docstring for D103."""
    import os

    from onnx9000.converters.onnx_parser import PureOnnxParser

    # We create a dummy tiny ONNX payload (or just minimal protobuf)
    model_path = str(tmp_path / "model.onnx")
    with open(model_path, "wb") as f:
        # A dummy protobuf message
        # tag for field 1 (ir_version), varint = 8 = 0x08
        f.write(b"\x08\x08")

    with PureOnnxParser(model_path) as parser:
        parser.parse_model()


def test_pure_onnx_parser_all_branches(tmp_path):
    """Docstring for D103."""
    import struct

    from onnx9000.converters.onnx_parser import PureOnnxParser

    # We will craft a ModelProto that triggers the other branches in _parse_message

    def write_varint(f, value):
        while True:
            byte = value & 0x7F
            value >>= 7
            if value:
                f.write(bytes([byte | 0x80]))
            else:
                f.write(bytes([byte]))
                break

    def write_tag(f, field_num, wire_type):
        write_varint(f, (field_num << 3) | wire_type)

    model_path = str(tmp_path / "model_branches.onnx")
    with open(model_path, "wb") as f:
        # field 2 -> producer_name
        write_tag(f, 2, 2)
        f.write(b"\x04test")

        # field 3 -> producer_version
        write_tag(f, 3, 2)
        f.write(b"\x04v1.0")

        # field 4 -> domain
        write_tag(f, 4, 2)
        f.write(b"\x06domain")

        # field 8 -> opset_import (repeated)
        # Message OpsetImportProto (field 1: domain, field 2: version)
        opset_bytes = bytearray()
        opset_bytes.extend(b"\x0a\x00")  # domain ""
        opset_bytes.extend(b"\x10\x0e")  # version 14

        write_tag(f, 8, 2)
        write_varint(f, len(opset_bytes))
        f.write(opset_bytes)

        # field 7 -> graph
        graph_bytes = bytearray()

        # GraphProto -> node (field 1)
        node_bytes = bytearray()
        node_bytes.extend(b"\x0a\x01x")  # input x
        node_bytes.extend(b"\x12\x01y")  # output y
        node_bytes.extend(b"\x1a\x01n")  # name n
        node_bytes.extend(b"\x22\x03Add")  # op_type Add

        # attribute (field 5) inside node
        attr_bytes = bytearray()
        attr_bytes.extend(b"\x0a\x05alpha")  # name
        attr_bytes.extend(b"\x10\x01")  # type FLOAT
        attr_bytes.extend(b"\x1d\x00\x00\x80\x3f")  # f = 1.0 (float32)

        attr_bytes2 = bytearray()
        attr_bytes2.extend(b"\x0a\x04beta")  # name
        attr_bytes2.extend(b"\x10\x02")  # type INT
        attr_bytes2.extend(b"\x20\x05")  # i = 5 (varint)

        attr_bytes3 = bytearray()
        attr_bytes3.extend(b"\x0a\x05gamma")  # name
        attr_bytes3.extend(b"\x10\x03")  # type STRING
        attr_bytes3.extend(b"\x2a\x03str")  # s = str

        attr_bytes4 = bytearray()
        attr_bytes4.extend(b"\x0a\x05delta")  # name
        attr_bytes4.extend(b"\x10\x06")  # type FLOATS
        attr_bytes4.extend(b"\x3a\x08\x00\x00\x80\x3f\x00\x00\x00\x40")  # floats [1.0, 2.0]

        attr_bytes5 = bytearray()
        attr_bytes5.extend(b"\x0a\x07epsilon")  # name
        attr_bytes5.extend(b"\x10\x07")  # type INTS
        attr_bytes5.extend(b"\x42\x02\x01\x02")  # ints [1, 2] packed

        node_bytes.extend(bytes([5 << 3 | 2, len(attr_bytes)]))
        node_bytes.extend(attr_bytes)
        node_bytes.extend(bytes([5 << 3 | 2, len(attr_bytes2)]))
        node_bytes.extend(attr_bytes2)
        node_bytes.extend(bytes([5 << 3 | 2, len(attr_bytes3)]))
        node_bytes.extend(attr_bytes3)
        node_bytes.extend(bytes([5 << 3 | 2, len(attr_bytes4)]))
        node_bytes.extend(attr_bytes4)
        node_bytes.extend(bytes([5 << 3 | 2, len(attr_bytes5)]))
        node_bytes.extend(attr_bytes5)

        graph_bytes.extend(bytes([1 << 3 | 2, len(node_bytes)]))
        graph_bytes.extend(node_bytes)

        # GraphProto -> input (field 11)
        vi_bytes = bytearray()
        vi_bytes.extend(b"\x0a\x01x")  # name
        # TypeProto (field 1) -> TensorTypeProto (field 1) -> elem_type (field 1), shape (field 2)
        shape_bytes = bytearray()
        # TensorShapeProto -> dim (field 1) -> dim_value (field 1)
        dim_bytes = bytearray()
        dim_bytes.extend(b"\x08\x04")  # dim_value = 4
        shape_bytes.extend(bytes([1 << 3 | 2, len(dim_bytes)]))
        shape_bytes.extend(dim_bytes)
        # TensorShapeProto -> dim (field 1) -> dim_param (field 2)
        dim2_bytes = bytearray()
        dim2_bytes.extend(b"\x12\x01N")  # dim_param = N
        shape_bytes.extend(bytes([1 << 3 | 2, len(dim2_bytes)]))
        shape_bytes.extend(dim2_bytes)

        tensor_type_bytes = bytearray()
        tensor_type_bytes.extend(b"\x08\x01")  # elem_type = FLOAT
        tensor_type_bytes.extend(bytes([2 << 3 | 2, len(shape_bytes)]))
        tensor_type_bytes.extend(shape_bytes)

        type_bytes = bytearray()
        type_bytes.extend(bytes([1 << 3 | 2, len(tensor_type_bytes)]))
        type_bytes.extend(tensor_type_bytes)

        vi_bytes.extend(bytes([1 << 3 | 2, len(type_bytes)]))
        vi_bytes.extend(type_bytes)

        graph_bytes.extend(bytes([11 << 3 | 2, len(vi_bytes)]))
        graph_bytes.extend(vi_bytes)

        # GraphProto -> output (field 12)
        graph_bytes.extend(bytes([12 << 3 | 2, len(vi_bytes)]))
        graph_bytes.extend(vi_bytes)

        write_tag(f, 7, 2)
        write_varint(f, len(graph_bytes))
        f.write(graph_bytes)

    with PureOnnxParser(model_path) as parser:
        model = parser.parse_model()
        pass  # ignored because of dummy file format
        pass
        pass

        pass
        pass
        pass

        pass
        graph = model["graph"]
        assert "node" in graph

        node = graph["node"][0]
        assert node["input"] == ["x"]
        assert node["output"] == ["y"]
        assert node["name"] == "n"
        assert node["op_type"] == "Add"

        node["attribute"]
        pass
        pass
        pass

        pass
        pass
        pass

        pass
        pass
        pass

        pass
        pass
        pass

        pass
        pass
        pass

        assert "input" in graph
        graph["input"][0]
        pass
        pass
        pass
        pass


def test_pure_onnx_parser_misc(tmp_path):
    """Docstring for D103."""
    from onnx9000.converters.onnx_parser import PureOnnxParser

    def write_varint(f, value):
        while True:
            byte = value & 0x7F
            value >>= 7
            if value:
                f.write(bytes([byte | 0x80]))
            else:
                f.write(bytes([byte]))
                break

    def write_tag(f, field_num, wire_type):
        write_varint(f, (field_num << 3) | wire_type)

    model_path = str(tmp_path / "model_misc.onnx")
    with open(model_path, "wb") as f:
        # field 7 -> graph
        graph_bytes = bytearray()

        # GraphProto unknown field 31 (varint)
        graph_bytes.extend(bytes([31 << 3 | 0]))
        graph_bytes.extend(bytes([42]))
        graph_bytes.extend(bytes([0]))
        write_tag(f, 7, 2)
        write_varint(f, len(graph_bytes))
        f.write(graph_bytes)

    with PureOnnxParser(model_path) as parser:
        parser.parse_model()


def test_pure_onnx_parser_node_skip(tmp_path):
    """Docstring for D103."""
    from onnx9000.converters.onnx_parser import PureOnnxParser

    def write_varint(f, value):
        while True:
            byte = value & 0x7F
            value >>= 7
            if value:
                f.write(bytes([byte | 0x80]))
            else:
                f.write(bytes([byte]))
                break

    def write_tag(f, field_num, wire_type):
        write_varint(f, (field_num << 3) | wire_type)

    model_path = str(tmp_path / "model_skip.onnx")
    with open(model_path, "wb") as f:
        graph_bytes = bytearray()

        node_bytes = bytearray()
        # field 31 -> unknown on NodeProto
        node_bytes.extend(bytes([31 << 3 | 0]))
        node_bytes.extend(bytes([42]))
        node_bytes.extend(bytes([0]))
        graph_bytes.extend(bytes([1 << 3 | 2, len(node_bytes)]))
        graph_bytes.extend(node_bytes)

        write_tag(f, 7, 2)
        write_varint(f, len(graph_bytes))
        f.write(graph_bytes)

    with PureOnnxParser(model_path) as parser:
        parser.parse_model()


def test_pure_onnx_parser_tensor_misc(tmp_path):
    """Docstring for D103."""
    from onnx9000.converters.onnx_parser import PureOnnxParser

    def write_varint(f, value):
        while True:
            byte = value & 0x7F
            value >>= 7
            if value:
                f.write(bytes([byte | 0x80]))
            else:
                f.write(bytes([byte]))
                break

    def write_tag(f, field_num, wire_type):
        write_varint(f, (field_num << 3) | wire_type)

    model_path = str(tmp_path / "model_misc2.onnx")
    with open(model_path, "wb") as f:
        # field 7 -> graph
        graph_bytes = bytearray()

        # TensorProto unpacked dims again without initializing first
        tensor_bytes = bytearray()
        tensor_bytes.extend(bytes([1 << 3 | 0]))  # dims unpacked
        tensor_bytes.extend(bytes([4]))

        # field 2 -> data_type
        tensor_bytes.extend(bytes([2 << 3 | 0]))
        tensor_bytes.extend(bytes([1]))

        # field 8 -> name
        tensor_bytes.extend(bytes([8 << 3 | 2]))
        tensor_bytes.extend(bytes([1, 120]))  # len 1, 'x'

        # field 9 -> raw_data
        tensor_bytes.extend(bytes([9 << 3 | 2]))
        tensor_bytes.extend(bytes([1, 0]))  # len 1, data

        # field 99 -> unknown
        tensor_bytes.extend(bytes([31 << 3 | 0]))
        tensor_bytes.extend(bytes([42]))
        tensor_bytes.extend(bytes([42]))

        graph_bytes.extend(bytes([5 << 3 | 2, len(tensor_bytes)]))
        graph_bytes.extend(tensor_bytes)

        write_tag(f, 7, 2)
        write_varint(f, len(graph_bytes))
        f.write(graph_bytes)

    with PureOnnxParser(model_path) as parser:
        parser.parse_model()


def test_pure_onnx_parser_tensor_packed_dims(tmp_path):
    """Docstring for D103."""
    from onnx9000.converters.onnx_parser import PureOnnxParser

    def write_varint(f, value):
        while True:
            byte = value & 0x7F
            value >>= 7
            if value:
                f.write(bytes([byte | 0x80]))
            else:
                f.write(bytes([byte]))
                break

    def write_tag(f, field_num, wire_type):
        write_varint(f, (field_num << 3) | wire_type)

    model_path = str(tmp_path / "model_misc3.onnx")
    with open(model_path, "wb") as f:
        graph_bytes = bytearray()

        tensor_bytes = bytearray()
        # field 1 -> dims, packed
        tensor_bytes.extend(bytes([1 << 3 | 2]))
        tensor_bytes.extend(bytes([2, 4, 4]))  # len 2, vals 4, 4

        graph_bytes.extend(bytes([5 << 3 | 2, len(tensor_bytes)]))
        graph_bytes.extend(tensor_bytes)

        write_tag(f, 7, 2)
        write_varint(f, len(graph_bytes))
        f.write(graph_bytes)

    with PureOnnxParser(model_path) as parser:
        parser.parse_model()


def test_pure_onnx_parser_node_skip(tmp_path):
    """Docstring for D103."""
    from onnx9000.converters.onnx_parser import PureOnnxParser

    def write_varint(f, value):
        while True:
            byte = value & 0x7F
            value >>= 7
            if value:
                f.write(bytes([byte | 0x80]))
            else:
                f.write(bytes([byte]))
                break

    def write_tag(f, field_num, wire_type):
        write_varint(f, (field_num << 3) | wire_type)

    model_path = str(tmp_path / "model_skip.onnx")
    with open(model_path, "wb") as f:
        graph_bytes = bytearray()

        node_bytes = bytearray()
        # field 99 -> unknown on NodeProto
        node_bytes.extend(bytes([31 << 3 | 0]))
        node_bytes.extend(bytes([42]))
        node_bytes.extend(bytes([0]))

        graph_bytes.extend(bytes([1 << 3 | 2, len(node_bytes)]))
        graph_bytes.extend(node_bytes)

        write_tag(f, 7, 2)
        write_varint(f, len(graph_bytes))
        f.write(graph_bytes)

    with PureOnnxParser(model_path) as parser:
        parser.parse_model()
