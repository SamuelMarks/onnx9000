import pytest
from pathlib import Path
import struct
import tempfile
from onnx9000.core.parser.core import parse_tensor_proto, parse_model, load_tensor
from onnx9000.core import onnx_pb2


def test_parse_tensor_proto_external_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        bin_path = base_dir / "external.bin"
        with open(bin_path, "wb") as f:
            f.write(b"0123456789")

        tp = onnx_pb2.TensorProto()
        tp.name = "ext_tensor"
        tp.data_type = 1  # FLOAT
        tp.dims.extend([2])
        tp.data_location = 1  # EXTERNAL

        entry_loc = tp.external_data.add()
        entry_loc.key = "location"
        entry_loc.value = "external.bin"

        entry_off = tp.external_data.add()
        entry_off.key = "offset"
        entry_off.value = "2"

        entry_len = tp.external_data.add()
        entry_len.key = "length"
        entry_len.value = "4"

        tensor = parse_tensor_proto(tp, base_dir)
        assert tensor.name == "ext_tensor"

        # Test offset only (no length)
        tp2 = onnx_pb2.TensorProto()
        tp2.name = "ext_tensor2"
        tp2.data_type = 1
        tp2.dims.extend([2])
        tp2.data_location = 1
        entry_loc2 = tp2.external_data.add()
        entry_loc2.key = "location"
        entry_loc2.value = "external.bin"
        entry_off2 = tp2.external_data.add()
        entry_off2.key = "offset"
        entry_off2.value = "2"

        tensor2 = parse_tensor_proto(tp2, base_dir)
        assert tensor2.name == "ext_tensor2"


def test_parse_tensor_proto_data_types():
    # int32
    tp_i32 = onnx_pb2.TensorProto()
    tp_i32.name = "t_i32"
    tp_i32.data_type = 6  # INT32
    tp_i32.dims.extend([2])
    tp_i32.int32_data.extend([42, -1])
    tensor_i32 = parse_tensor_proto(tp_i32)
    assert tensor_i32.name == "t_i32"

    # int64
    tp_i64 = onnx_pb2.TensorProto()
    tp_i64.name = "t_i64"
    tp_i64.data_type = 7  # INT64
    tp_i64.dims.extend([2])
    tp_i64.int64_data.extend([4200000000, -1])
    tensor_i64 = parse_tensor_proto(tp_i64)
    assert tensor_i64.name == "t_i64"

    # string
    tp_str = onnx_pb2.TensorProto()
    tp_str.name = "t_str"
    tp_str.data_type = 8  # STRING
    tp_str.dims.extend([2])
    tp_str.string_data.extend([b"hello", b"world"])
    tensor_str = parse_tensor_proto(tp_str)
    assert tensor_str.name == "t_str"


def test_parse_model_metadata():
    mp = onnx_pb2.ModelProto()
    gp = mp.graph
    gp.name = "test_graph"
    mp.producer_name = "test_producer"

    prop = mp.metadata_props.add()
    prop.key = "my_key"
    prop.value = "my_value"

    graph = parse_model(mp)
    assert graph.metadata_props["my_key"] == "my_value"


def test_load_tensor():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "tensor.pb"
        tp = onnx_pb2.TensorProto()
        tp.name = "t1"
        tp.data_type = 1
        tp.dims.extend([1])
        tp.float_data.extend([1.0])
        with open(path, "wb") as f:
            f.write(tp.SerializeToString())

        tensor = load_tensor(path)
        assert tensor.name == "t1"
