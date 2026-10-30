"""Unit tests for ONNX protobuf parsing in `onnx9000.core.parser.core`."""

import tempfile

import pytest
from onnx9000.core import onnx_pb2
from onnx9000.core.dtypes import DType
from onnx9000.core.exceptions import ONNXParseError
from onnx9000.core.ir import DynamicDim, Graph, Node, ValueInfo
from onnx9000.core.parser.core import (
    _parse_attribute,
    _parse_dtype,
    _parse_shape,
    from_bytes,
    load,
    load_tensor,
    parse_model,
    parse_sparse_tensor_proto,
    parse_tensor_proto,
)
from onnx9000.core.serializer import to_bytes


def test_parser_core_functions() -> None:
    """Verify _parse_dtype, _parse_shape, _parse_attribute, parse_tensor_proto, parse_model, load, and from_bytes."""
    assert _parse_dtype(onnx_pb2.TensorProto.FLOAT) == DType.FLOAT32
    with pytest.raises(ONNXParseError):
        _parse_dtype(99999)

    shape_proto = onnx_pb2.TensorShapeProto()
    d0 = shape_proto.dim.add()
    d0.dim_value = 4
    d1 = shape_proto.dim.add()
    d1.dim_param = "N"
    assert _parse_shape(shape_proto) == (4, DynamicDim("N"))

    attr_proto = onnx_pb2.AttributeProto()
    attr_proto.name = "alpha"
    attr_proto.type = onnx_pb2.AttributeProto.FLOAT
    attr_proto.f = 0.25
    parsed_attr = _parse_attribute(attr_proto)
    assert parsed_attr.name == "alpha"
    assert parsed_attr.value == pytest.approx(0.25)

    tp = onnx_pb2.TensorProto()
    tp.name = "w"
    tp.data_type = onnx_pb2.TensorProto.FLOAT
    tp.dims.extend([2])
    tp.float_data.extend([1.0, 2.0])
    parsed_t = parse_tensor_proto(tp)
    assert parsed_t.name == "w"
    assert parsed_t.shape == (2,)

    with tempfile.NamedTemporaryFile(suffix=".pb") as tf:
        tf.write(tp.SerializeToString())
        tf.flush()
        loaded_t = load_tensor(tf.name)
        assert loaded_t.name == "w"

    sp = onnx_pb2.SparseTensorProto()
    sp.values.CopyFrom(tp)
    idx_tp = onnx_pb2.TensorProto()
    idx_tp.name = "w_idx"
    idx_tp.data_type = onnx_pb2.TensorProto.INT64
    idx_tp.dims.extend([2])
    idx_tp.int64_data.extend([0, 1])
    sp.indices.CopyFrom(idx_tp)
    sp.dims.extend([2])
    parsed_sp = parse_sparse_tensor_proto(sp)
    assert parsed_sp.shape == (2,)

    g = Graph("roundtrip")
    g.inputs.append(ValueInfo("x", (2,), DType.FLOAT32))
    g.outputs.append(ValueInfo("y", (2,), DType.FLOAT32))
    g.add_node(Node("Relu", ["x"], ["y"], name="r0"))
    raw_model = to_bytes(g)

    g_from_bytes = from_bytes(raw_model)
    assert g_from_bytes.name == "roundtrip"
    assert len(g_from_bytes.nodes) == 1

    mp = onnx_pb2.ModelProto()
    mp.ParseFromString(raw_model)
    g_parsed = parse_model(mp)
    assert g_parsed.name == "roundtrip"

    with tempfile.NamedTemporaryFile(suffix=".onnx") as mf:
        mf.write(raw_model)
        mf.flush()
        g_loaded = load(mf.name)
        assert g_loaded.name == "roundtrip"
