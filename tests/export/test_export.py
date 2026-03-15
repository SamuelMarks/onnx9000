"""Module providing core logic and structural definitions."""

import pytest
from pathlib import Path
import numpy as np
from onnx9000.core import onnx_pb2
from onnx9000.frontends.frontend.builder import GraphBuilder
from onnx9000.frontends.frontend.tensor import Tensor, Parameter, Node
from onnx9000.core.dtypes import DType
from onnx9000.core.exceptions import CompilationError
from onnx9000.export.builder import (
    build_graph_proto,
    build_model_proto,
    validate_model,
    sanitize_model,
    to_string,
    to_onnx,
)
from onnx9000.export.bundle import create_model_bundle
from onnx9000.export.chunking import (
    export_with_external_data,
    generate_chunk_manifest,
    compress_weights_to_int8,
    embed_metadata,
    calculate_sha256,
    merge_chunks,
    encrypt_chunk,
    decrypt_chunk,
)
from onnx9000.export.proto_utils import (
    to_tensor_proto,
    to_value_info_proto,
    to_attribute_proto,
    to_node_proto,
)


def test_builder(tmp_path):
    """Provides semantic functionality and verification."""
    gb = GraphBuilder("test_graph")
    t_in = Tensor((1, 10), DType.FLOAT32, "in")
    p1 = Parameter(
        (10, 10), DType.FLOAT32, "p1", data=np.ones((10, 10), dtype=np.float32)
    )
    p2 = Parameter((10,), DType.FLOAT32, "p2")
    p2.data = np.zeros(10, dtype=np.float32)
    out = Tensor((1, 10), DType.FLOAT32, "out")
    n = Node("MatMul", [t_in, p1], [out], {"alpha": 1.0})
    gb.inputs = [t_in]
    gb.outputs = [out]
    gb.nodes = [n]
    graph_proto = build_graph_proto(gb)
    assert graph_proto.name == "test_graph"
    model_proto = build_model_proto(gb)
    assert model_proto.producer_name == "onnx9000"
    validate_model(model_proto)
    sanitize_model(model_proto)
    s = to_string(gb)
    assert isinstance(s, bytes)
    file_path = tmp_path / "model.onnx"
    to_onnx(gb, file_path)
    assert file_path.exists()
    bad_model = onnx_pb2.ModelProto()
    with pytest.raises(CompilationError):
        validate_model(bad_model)


def test_builder_strings(tmp_path):
    """Provides semantic functionality and verification."""
    gb = GraphBuilder("test_graph")
    t_in = Tensor((1, 10), DType.FLOAT32, "in")
    n = Node("Relu", ["in"], ["out_str"], {})
    gb.inputs = [t_in]
    gb.outputs = [Tensor((1, 10), DType.FLOAT32, "final_out")]
    n2 = Node("Relu", ["out_str"], ["final_out"], {})
    gb.nodes.append(n2)
    gb.nodes = [n]
    graph_proto = build_graph_proto(gb)
    assert len(graph_proto.value_info) > 0


def test_proto_utils_tensor():
    """Provides semantic functionality and verification."""
    p = Parameter((10, "dyn"), DType.FLOAT32, "p1", data=np.ones(10, dtype=np.float32))
    tp = to_tensor_proto(p)
    assert tp.name == "p1"
    p_f32 = Parameter((2,), DType.FLOAT32, "p2", data=[1.0, 2.0])
    tp_f32 = to_tensor_proto(p_f32)
    p_f64 = Parameter((2,), DType.FLOAT64, "p_f64", data=[1.0, 2.0])
    tp_f64 = to_tensor_proto(p_f64)
    p_i32 = Parameter((2,), DType.INT32, "p_i32", data=[1, 2])
    tp_i32 = to_tensor_proto(p_i32)
    p_i64 = Parameter((2,), DType.INT64, "p_i64", data=[1, 2])
    tp_i64 = to_tensor_proto(p_i64)
    p_i16 = Parameter((2,), DType.INT16, "p_i16", data=[1, 2])
    to_tensor_proto(p_i16)
    p_i8 = Parameter((2,), DType.INT8, "p_i8", data=[1, 2])
    to_tensor_proto(p_i8)
    p_u8 = Parameter((2,), DType.UINT8, "p_u8", data=[1, 2])
    to_tensor_proto(p_u8)
    p_u16 = Parameter((2,), DType.UINT16, "p_u16", data=[1, 2])
    to_tensor_proto(p_u16)
    p_u32 = Parameter((2,), DType.UINT32, "p_u32", data=[1, 2])
    to_tensor_proto(p_u32)
    p_u64 = Parameter((2,), DType.UINT64, "p_u64", data=[1, 2])
    to_tensor_proto(p_u64)
    p_b = Parameter((2,), DType.BOOL, "p_b", data=[True, False])
    to_tensor_proto(p_b)


def test_proto_utils_attr():
    """Provides semantic functionality and verification."""
    a1 = to_attribute_proto("f", 1.0)
    assert a1.f == 1.0
    a2 = to_attribute_proto("i", 1)
    assert a2.i == 1
    a3 = to_attribute_proto("s", "test")
    assert a3.s == b"test"
    a4 = to_attribute_proto("ints", [1, 2])
    assert list(a4.ints) == [1, 2]
    a5 = to_attribute_proto("floats", [1.0, 2.0])
    assert list(a5.floats) == [1.0, 2.0]
    a6 = to_attribute_proto("strings", ["a", "b"])
    assert list(a6.strings) == [b"a", b"b"]
    with pytest.raises(ValueError):
        to_attribute_proto("bad_list", [object()])
    with pytest.raises(ValueError):
        to_attribute_proto("bad_type", object())


def test_bundle(tmp_path):
    """Provides semantic functionality and verification."""
    onnx_f = tmp_path / "model.onnx"
    onnx_f.write_bytes(b"mock")
    manifest_f = tmp_path / "manifest.json"
    manifest_f.write_text("{}")
    ext_dir = tmp_path / "ext"
    ext_dir.mkdir()
    (ext_dir / "w.bin").write_bytes(b"weights")
    out_z = tmp_path / "bundle.o9k"
    create_model_bundle(out_z, onnx_f, ext_dir, manifest_f)
    assert out_z.exists()


def test_chunking(tmp_path):
    """Provides semantic functionality and verification."""
    f = tmp_path / "test.txt"
    f.write_bytes(b"hello world")
    h = calculate_sha256(f)
    assert isinstance(h, str)
    assert len(h) == 64
    g = GraphBuilder("mock")
    export_with_external_data(g, tmp_path)
    generate_chunk_manifest(g, {})
    compress_weights_to_int8(g)
    embed_metadata(g, "a", "1", "txt")
    merge_chunks(Path("m"), Path("o"))
    encrypt_chunk(Path("c"), b"key")
    decrypt_chunk(Path("c"), b"key")


def test_builder_intermediate_tensor():
    """Provides semantic functionality and verification."""
    gb = GraphBuilder("mock")
    t_in = Tensor((1, 10), DType.FLOAT32, "in")
    t_mid = Tensor((1, 10), DType.FLOAT32, "mid")
    t_out = Tensor((1, 10), DType.FLOAT32, "out")
    n1 = Node("Relu", [t_in], [t_mid], {})
    n2 = Node("Relu", [t_mid], [t_out], {})
    gb.inputs = [t_in]
    gb.outputs = [t_out]
    gb.nodes = [n1, n2]
    build_graph_proto(gb)


def test_builder_param_no_data():
    """Provides semantic functionality and verification."""
    gb = GraphBuilder("mock")
    t_in = Tensor((1, 10), DType.FLOAT32, "in")
    p = Parameter((10,), DType.FLOAT32, "p")
    p.data = None
    t_out = Tensor((1, 10), DType.FLOAT32, "out")
    n1 = Node("Add", [t_in, p], [t_out], {})
    gb.inputs = [t_in]
    gb.outputs = [t_out]
    gb.nodes = [n1]
    build_graph_proto(gb)
