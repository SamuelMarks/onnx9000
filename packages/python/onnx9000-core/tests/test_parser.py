import os
import struct
import tempfile

import pytest
from onnx9000.core import onnx_pb2
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, DynamicDim, Graph, Node, Tensor, ValueInfo
from onnx9000.core.parser.core import _parse_dtype, from_bytes, load, load_tensor, parse_model
from onnx9000.core.serializer import save, serialize_model


def test_parse_dtype() -> None:
    assert _parse_dtype(onnx_pb2.TensorProto.FLOAT) == DType.FLOAT32


def test_parser_core() -> None:
    g = Graph("my_graph")
    g.doc_string = "My doc"
    g.opset_imports["ai.onnx"] = 14
    t_float = Tensor("f1", (1,), DType.FLOAT32, data=struct.pack("<f", 1.0))
    g.add_tensor(t_float)
    g.initializers.append("f1")
    t_i32 = Tensor("i1", (1,), DType.INT32, data=struct.pack("<i", 1))
    g.add_tensor(t_i32)
    g.initializers.append("i1")
    t_i64 = Tensor("i64", (1,), DType.INT64, data=struct.pack("<q", 1))
    g.add_tensor(t_i64)
    g.initializers.append("i64")
    t_bool = Tensor("b1", (1,), DType.BOOL, data=struct.pack("<?", True))
    g.add_tensor(t_bool)
    g.initializers.append("b1")
    t_str = Tensor("s1", (1,), DType.STRING, data=b"hello")
    g.add_tensor(t_str)
    g.initializers.append("s1")
    g.inputs.append("f1")
    g.outputs.append("i1")
    tp = onnx_pb2.TensorProto()
    gp = onnx_pb2.GraphProto()
    attr_dict = {
        "a_float": Attribute("a_float", "FLOAT", 1.0),
        "a_int": Attribute("a_int", "INT", 1),
        "a_string": Attribute("a_string", "STRING", "1"),
        "a_floats": Attribute("a_floats", "FLOATS", [1.0]),
        "a_ints": Attribute("a_ints", "INTS", [1]),
        "a_strings": Attribute("a_strings", "STRINGS", ["1"]),
        "a_tensor": Attribute("a_tensor", "TENSOR", tp),
        "a_graph": Attribute("a_graph", "GRAPH", gp),
    }
    g.add_node(Node("TestOp", ["f1"], ["i1"], attributes=attr_dict, name="N1", domain="ai.onnx"))
    with tempfile.NamedTemporaryFile(delete=False) as f:
        file_path = f.name
    save(g, file_path)
    g_loaded = load(file_path)
    assert g_loaded.name == "my_graph"
    assert g_loaded.doc_string == "My doc"
    assert "f1" in g_loaded.initializers
    with open(file_path, "rb") as f:
        raw_bytes = f.read()
    g_loaded_2 = from_bytes(raw_bytes)
    assert g_loaded_2.name == "my_graph"
    os.remove(file_path)


def test_load_tensor() -> None:
    Tensor("f1", (1,), DType.FLOAT32, data=struct.pack("<f", 1.0))
    proto = onnx_pb2.TensorProto()
    proto.name = "f1"
    proto.data_type = onnx_pb2.TensorProto.FLOAT
    proto.dims.extend([1])
    proto.raw_data = struct.pack("<f", 1.0)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(proto.SerializeToString())
        file_path = f.name
    t_loaded = load_tensor(file_path)
    assert t_loaded.name == "f1"
    os.remove(file_path)


def test_parser_unsupported_dtype() -> None:
    from onnx9000.core.exceptions import ONNXParseError

    with pytest.raises(ONNXParseError):
        _parse_dtype(9999)


def test_parser_other_data_fields() -> None:
    tp_float = onnx_pb2.TensorProto(
        name="f2", data_type=onnx_pb2.TensorProto.FLOAT, dims=[1], float_data=[1.0]
    )
    tp_i32 = onnx_pb2.TensorProto(
        name="i2", data_type=onnx_pb2.TensorProto.INT32, dims=[1], int32_data=[1]
    )
    tp_i64 = onnx_pb2.TensorProto(
        name="i64_2", data_type=onnx_pb2.TensorProto.INT64, dims=[1], int64_data=[1]
    )
    tp_str = onnx_pb2.TensorProto(
        name="s2", data_type=onnx_pb2.TensorProto.STRING, dims=[1], string_data=[b"str"]
    )
    model = onnx_pb2.ModelProto()
    model.graph.name = "g"
    model.graph.initializer.extend([tp_float, tp_i32, tp_i64, tp_str])
    g = parse_model(model)
    assert "f2" in g.tensors
    assert "i2" in g.tensors
    assert "i64_2" in g.tensors
    assert "s2" in g.tensors


def test_parser_uncovered_paths() -> None:
    g = Graph("g")
    g.add_node(Node("N", [], [], {"bad_attr": Attribute("bad_attr", "UNKNOWN", None)}))
    v_in = ValueInfo("v1", (DynamicDim("N"), DynamicDim(-1), 1), DType.FLOAT32)
    v_out = ValueInfo("v2", (1,), DType.FLOAT32)
    g.inputs.append(v_in)
    g.outputs.append(v_out)
    ValueInfo("v_mid", (1,), DType.FLOAT32)
    model_proto = serialize_model(g)
    vi_proto = onnx_pb2.ValueInfoProto()
    vi_proto.name = "v_mid"
    vi_proto.type.tensor_type.elem_type = onnx_pb2.TensorProto.FLOAT
    vi_proto.type.tensor_type.shape.dim.add().dim_value = 1
    vi_proto.type.tensor_type.shape.dim.add()
    model_proto.graph.value_info.append(vi_proto)
    n_proto = model_proto.graph.node.add()
    n_proto.op_type = "StrangeOp"
    n_proto.name = "S1"
    attr_proto = n_proto.attribute.add()
    attr_proto.name = "strange"
    attr_proto.type = 0
    g2 = parse_model(model_proto)
    assert "v_mid" in g2.tensors
    assert len(g2.inputs) == 1
    assert "v1" in g2.tensors
    assert g2.tensors["v1"].shape[0].value == "N"
    assert g2.tensors["v1"].shape[1] == -1
    assert "v2" in g2.tensors
