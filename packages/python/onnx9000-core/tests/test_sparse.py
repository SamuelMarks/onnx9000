"""Tests for sparse."""

import json
import logging
import os
import struct

import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Constant, Graph, Node, SparseTensor
from onnx9000.core.sparse import (
    analyze_topological_dead_ends,
    calculate_memory_usage,
    collapse_sparse_tensors,
    compression_ratio,
    convert_hf_sparse_to_onnx,
    de_sparsify,
    dense_to_bsr,
    dense_to_coo,
    dense_to_csc,
    dense_to_csr,
    detect_theoretical_sparsity,
    evaluate_mse,
    generate_json_report,
    get_byte_size,
    get_sparsity_report,
    get_struct_fmt,
    map_sparse_to_safetensors,
    pack_data,
    pack_sparse_int8,
    profile,
    resolve_nm_metadata,
    sparse_to_coo,
    sparse_to_dense,
    strip_dense_representation,
    unpack_data,
    validate_provider_support,
)


def test_get_struct_fmt():
    """Docstring for D103."""
    pack_data([1.0], DType.FLOAT16)
    unpack_data(struct.pack("<e", 1.0), DType.FLOAT16)

    assert get_struct_fmt(DType.FLOAT32) == "f"
    assert get_struct_fmt(DType.FLOAT64) == "d"
    assert get_struct_fmt(DType.INT8) == "b"
    assert get_struct_fmt(DType.INT16) == "h"
    assert get_struct_fmt(DType.INT32) == "i"
    assert get_struct_fmt(DType.INT64) == "q"
    assert get_struct_fmt(DType.UINT8) == "B"
    assert get_struct_fmt(DType.UINT16) == "H"
    assert get_struct_fmt(DType.UINT32) == "I"
    assert get_struct_fmt(DType.UINT64) == "Q"
    assert get_struct_fmt(DType.BOOL) == "?"
    assert get_struct_fmt(DType.FLOAT16) == "e"

    # Simulate ValueError/AttributeError
    assert get_struct_fmt("UnknownDType") == "B"

    class BadType:
        """Bad type."""

        def __eq__(self, other):
            """Eq."""
            return False

        def __hash__(self):
            """Hash."""
            return 1

    assert get_struct_fmt(BadType()) == "B"


def test_get_byte_size():
    """Docstring for D103."""
    assert get_byte_size(DType.FLOAT32) == 4
    assert get_byte_size(DType.INT64) == 8


def test_unpack_data():
    """Docstring for D103."""
    assert unpack_data(b"", DType.FLOAT32) == []
    data = struct.pack("<2f", 1.0, 2.0)
    assert unpack_data(data, DType.FLOAT32) == [1.0, 2.0]


def test_pack_data():
    """Docstring for D103."""
    assert pack_data([], DType.FLOAT32) == b""
    data = pack_data([1.0, 2.0], DType.FLOAT32)
    assert struct.unpack("<2f", data) == (1.0, 2.0)
    # Test float64 downcast
    data64 = pack_data([1.0, 2.0], DType.FLOAT64)
    assert struct.unpack("<2f", data64) == (1.0, 2.0)


def test_pack_sparse_int8():
    """Docstring for D103."""
    values = [0xA, 0xB, 0xC]
    indices = [1, 2, 3]
    packed = pack_sparse_int8(values, indices)
    # Expected:
    # i=0: idx1=1, val1=10(A), idx2=2, val2=11(B)
    # b1 = (1<<4)|(10>>4) = 16|0 = 16 (0x10)
    # b2 = ((10&15)<<4)|2 = 160|2 = 162 (0xA2)
    # b3 = 11 = 0x0B
    # i=2: idx1=3, val1=12(C)
    # b1 = (3<<4)|(12>>4) = 48|0 = 48 (0x30)
    # b2 = (12&15)<<4 = 12<<4 = 192 (0xC0)
    assert packed == bytes([0x10, 0xA2, 0x0B, 0x30, 0xC0])


def test_dense_to_coo():
    """Docstring for D103."""
    c1 = Constant("c1", shape=(2, 2), dtype=DType.FLOAT32)
    c1.data = pack_data([0.0, 1.0, 0.0, 2.0], DType.FLOAT32)
    coo = dense_to_coo(c1)
    assert coo.format == "COO"
    assert unpack_data(coo.values.data, coo.values.dtype) == [1.0, 2.0]
    assert unpack_data(coo.indices.data, coo.indices.dtype) == [1, 3]

    c_empty = Constant("ce", shape=(2,), dtype=DType.FLOAT32)
    coo_empty = dense_to_coo(c_empty)
    assert coo_empty.values is None
    assert coo_empty.indices is None


def test_dense_to_csr():
    """Docstring for D103."""
    c1 = Constant("c1", shape=(2, 2), dtype=DType.FLOAT32)
    c1.data = pack_data([0.0, 1.0, 2.0, 0.0], DType.FLOAT32)
    csr = dense_to_csr(c1)
    assert csr.format == "CSR"
    assert unpack_data(csr.values.data, csr.values.dtype) == [1.0, 2.0]
    assert unpack_data(csr.col_indices.data, csr.col_indices.dtype) == [1, 0]
    assert unpack_data(csr.row_ptr.data, csr.row_ptr.dtype) == [0, 1, 2]

    # Non-2D falls back to COO
    c2 = Constant("c2", shape=(2,), dtype=DType.FLOAT32)
    c2.data = pack_data([0.0, 1.0], DType.FLOAT32)
    csr_fallback = dense_to_csr(c2)
    assert csr_fallback.format == "COO"


def test_dense_to_csr_large_dims():
    """Docstring for D103."""
    c1 = Constant("c1", shape=(2**32, 2), dtype=DType.FLOAT32)
    with pytest.raises(ValueError):
        dense_to_csr(c1)


def test_dense_to_csc():
    """Docstring for D103."""
    c1 = Constant("c1", shape=(2, 2), dtype=DType.FLOAT32)
    c1.data = pack_data([0.0, 1.0, 2.0, 0.0], DType.FLOAT32)
    csc = dense_to_csc(c1)
    assert csc.format == "CSC"
    assert unpack_data(csc.values.data, csc.values.dtype) == [2.0, 1.0]
    assert unpack_data(csc.col_indices.data, csc.col_indices.dtype) == [1, 0]
    assert unpack_data(csc.row_ptr.data, csc.row_ptr.dtype) == [0, 1, 2]

    c2 = Constant("c2", shape=(2,), dtype=DType.FLOAT32)
    csc_fallback = dense_to_csc(c2)
    assert csc_fallback.format == "COO"


def test_dense_to_bsr():
    """Docstring for D103."""
    c1 = Constant("c1", shape=(2, 2), dtype=DType.FLOAT32)
    c1.data = pack_data([0.0, 1.0, 2.0, 0.0], DType.FLOAT32)
    bsr = dense_to_bsr(c1, (1, 1))
    assert bsr.format == "BSR"
    assert unpack_data(bsr.values.data, bsr.values.dtype) == [1.0, 2.0]
    assert unpack_data(bsr.col_indices.data, bsr.col_indices.dtype) == [1, 0]
    assert unpack_data(bsr.row_ptr.data, bsr.row_ptr.dtype) == [0, 1, 2]

    # Invalid blocks
    c2 = Constant("c2", shape=(2, 2), dtype=DType.FLOAT32)
    bsr2 = dense_to_bsr(c2, (3, 3))
    assert bsr2.format == "COO"

    # Not 2D
    c3 = Constant("c3", shape=(2,), dtype=DType.FLOAT32)
    bsr3 = dense_to_bsr(c3, (1, 1))
    assert bsr3.format == "COO"


def test_sparse_to_coo():
    """Docstring for D103."""
    c1 = Constant("c1", shape=(2, 2), dtype=DType.FLOAT32)
    c1.data = pack_data([0.0, 1.0, 2.0, 0.0], DType.FLOAT32)

    coo = dense_to_coo(c1)
    assert sparse_to_coo(coo) is coo

    csr = dense_to_csr(c1)
    coo_from_csr = sparse_to_coo(csr)
    assert coo_from_csr.format == "COO"
    assert unpack_data(coo_from_csr.values.data, DType.FLOAT32) == [1.0, 2.0]
    assert unpack_data(coo_from_csr.indices.data, DType.INT64) == [1, 2]

    csc = dense_to_csc(c1)
    coo_from_csc = sparse_to_coo(csc)
    assert coo_from_csc.format == "COO"
    assert unpack_data(coo_from_csc.values.data, DType.FLOAT32) == [2.0, 1.0]
    assert unpack_data(coo_from_csc.indices.data, DType.INT64) == [2, 1]

    bsr = dense_to_bsr(c1, (1, 1))
    coo_from_bsr = sparse_to_coo(bsr)
    assert coo_from_bsr.format == "COO"
    assert unpack_data(coo_from_bsr.values.data, DType.FLOAT32) == [1.0, 2.0]
    assert unpack_data(coo_from_bsr.indices.data, DType.INT64) == [1, 2]

    # Unknown
    s = SparseTensor("s", format="UNK", dims=(2,))
    assert sparse_to_coo(s) is s


def test_sparse_to_dense():
    """Docstring for D103."""
    c1 = Constant("c1", shape=(2, 2), dtype=DType.FLOAT32)
    c1.data = pack_data([0.0, 1.0, 2.0, 0.0], DType.FLOAT32)
    coo = dense_to_coo(c1)
    dense = sparse_to_dense(coo)
    assert dense.name == "c1"
    assert unpack_data(dense.data, dense.dtype) == [0.0, 1.0, 2.0, 0.0]

    # Empty
    ce = Constant("ce", shape=(2, 2), dtype=DType.FLOAT32)
    cooe = dense_to_coo(ce)
    densee = sparse_to_dense(cooe)
    assert densee.shape == (2, 2)
    assert densee.dtype == DType.FLOAT32


def test_detect_theoretical_sparsity():
    """Docstring for D103."""
    c1 = Constant("c1", shape=(4,), dtype=DType.FLOAT32)
    c1.data = pack_data([0.0, 0.0, 1.0, 1.0], DType.FLOAT32)
    assert detect_theoretical_sparsity(c1) == 0.5

    ce = Constant("ce", shape=(4,), dtype=DType.FLOAT32)
    assert detect_theoretical_sparsity(ce) == 1.0
    ce.data = b""
    assert detect_theoretical_sparsity(ce) == 1.0


def test_calculate_memory_usage():
    """Docstring for D103."""
    empty_s = SparseTensor("es", format="CSR", dims=(4,))
    empty_s.row_ptr = Constant("r")
    empty_s.col_indices = Constant("c")
    assert calculate_memory_usage(empty_s) == 0

    c1 = Constant("c1", shape=(4,), dtype=DType.FLOAT32)
    c1.data = pack_data([0.0, 0.0, 1.0, 1.0], DType.FLOAT32)
    assert calculate_memory_usage(c1) == 16
    ce = Constant("ce", shape=(4,), dtype=DType.FLOAT32)
    assert calculate_memory_usage(ce) == 0

    coo = dense_to_coo(c1)
    assert calculate_memory_usage(coo) == 8 + 16  # 2 floats, 2 int64


def test_compression_ratio():
    """Docstring for D103."""
    c1 = Constant("c1", shape=(4,), dtype=DType.FLOAT32)
    c1.data = pack_data([0.0, 0.0, 1.0, 1.0], DType.FLOAT32)
    coo = dense_to_coo(c1)
    assert compression_ratio(c1, coo) == 1.0 - (24 / 16.0)

    ce = Constant("ce", shape=(4,), dtype=DType.FLOAT32)
    assert compression_ratio(ce, coo) == 0.0


def test_profile():
    """Docstring for D103."""
    g = Graph("g")
    assert profile(g) == {
        "layers": [],
        "total_dense_size": 0,
        "total_sparse_size": 0,
        "overall_sparsity": 0.0,
    }

    c1 = Constant("c1", shape=(4,), dtype=DType.FLOAT32)
    c1.data = pack_data([0.0, 0.0, 1.0, 1.0], DType.FLOAT32)
    coo = dense_to_coo(c1)
    g.tensors["c1"] = c1
    g.tensors["coo"] = coo
    rep = profile(g)
    assert len(rep["layers"]) == 2
    assert rep["overall_sparsity"] < 0


def test_get_sparsity_report():
    """Docstring for D103."""
    g = Graph("g")
    assert get_sparsity_report(g) == "No trainable parameters found."

    c1 = Constant("c1", shape=(4,), dtype=DType.FLOAT32)
    c1.data = pack_data([0.0, 0.0, 0.0, 0.0], DType.FLOAT32)
    g.tensors["c1"] = c1
    report = get_sparsity_report(g)
    assert "OVERALL" in report


def test_generate_json_report(tmp_path):
    """Docstring for D103."""
    g = Graph("g")
    p = os.path.join(tmp_path, "report.json")
    generate_json_report(g, p)
    assert os.path.exists(p)
    with open(p) as f:
        d = json.load(f)
    assert "layers" in d


def test_evaluate_mse():
    """Docstring for D103."""
    c1 = Constant("c1", shape=(4,), dtype=DType.FLOAT32)
    c1.data = pack_data([1.0, 1.0, 1.0, 1.0], DType.FLOAT32)

    c2 = Constant("c2", shape=(4,), dtype=DType.FLOAT32)
    c2.data = pack_data([1.0, 1.0, 0.0, 0.0], DType.FLOAT32)

    assert evaluate_mse(c1, c2) == 0.5

    coo = dense_to_coo(c2)
    assert evaluate_mse(c1, coo) == 0.5

    ce = Constant("ce", shape=(4,), dtype=DType.FLOAT32)
    assert evaluate_mse(ce, c2) == 0.0


def test_de_sparsify():
    """Docstring for D103."""
    g = Graph("g")
    c1 = Constant("c1", shape=(4,), dtype=DType.FLOAT32)
    c1.data = pack_data([1.0, 0.0, 0.0, 1.0], DType.FLOAT32)
    coo = dense_to_coo(c1)
    g.tensors["s1"] = coo
    g.sparse_initializers.append("s1")
    de_sparsify(g)
    assert isinstance(g.tensors["s1"], Constant)
    assert "s1" not in g.sparse_initializers
    assert "s1" in g.initializers


def test_convert_hf_sparse_to_onnx():
    """Docstring for D103."""
    g = Graph("g")
    c1 = Constant("c1", shape=(4,), dtype=DType.FLOAT32)
    c1.data = pack_data([1.0, 0.0, 0.0, 1.0], DType.FLOAT32)
    c1.metadata_props = {"sparse_method": "hf"}
    g.tensors["c1"] = c1
    g.initializers.append("c1")
    convert_hf_sparse_to_onnx(g)
    assert isinstance(g.tensors["c1"], SparseTensor)
    assert "c1" not in g.initializers
    assert "c1" in g.sparse_initializers


def test_resolve_nm_metadata():
    """Docstring for D103."""
    g = Graph("g")
    s1 = SparseTensor("s1", format="NM", dims=(2, 2))
    g.tensors["s1"] = s1
    resolve_nm_metadata(g)
    assert s1.metadata_props["sparse_encoding"] == "2:4"


def test_map_sparse_to_safetensors():
    """Docstring for D103."""
    g = Graph("g")
    val_const = Constant("v", shape=(1,), dtype=DType.FLOAT32)
    idx_const = Constant("i", shape=(1,), dtype=DType.INT64)
    s1 = SparseTensor("s1", format="COO", dims=(2, 2), values=val_const, indices=idx_const)
    g.tensors["s1"] = s1
    map_sparse_to_safetensors(g)
    assert s1.metadata_props["safetensors_reference"] == "sparse/s1"
    assert s1.values.metadata_props["safetensors_reference"] == "sparse/s1/values"
    assert s1.indices.metadata_props["safetensors_reference"] == "sparse/s1/indices"


def test_validate_provider_support(caplog):
    """Docstring for D103."""
    g = Graph("g")
    s1 = SparseTensor("s1", format="COO", dims=(2, 2))
    g.tensors["s1"] = s1
    with caplog.at_level(logging.WARNING):
        validate_provider_support(g, ["CUDA", "WebGPU"])
    assert "CUDA" in caplog.text


def test_collapse_sparse_tensors():
    """Docstring for D103."""
    g = Graph("g")
    c1 = Constant("c1", shape=(4,), dtype=DType.FLOAT32)
    c1.data = pack_data([0.0, 0.0, 0.0, 0.0], DType.FLOAT32)
    g.tensors["c1"] = c1

    s1 = SparseTensor("s1", format="COO", dims=(4,))
    s1.dtype = DType.FLOAT32

    g.tensors["s1"] = s1
    g.sparse_initializers.append("s1")

    val_const = Constant("v", shape=(4,), dtype=DType.FLOAT32)
    val_const.data = pack_data([0.0, 0.0, 0.0, 0.0], DType.FLOAT32)
    s2 = SparseTensor("s2", format="COO", dims=(4,), values=val_const)
    s2.dtype = DType.FLOAT32
    g.tensors["s2"] = s2

    collapse_sparse_tensors(g)

    assert isinstance(g.tensors["c1"], Constant)
    assert unpack_data(g.tensors["c1"].data, DType.FLOAT32) == [0.0]

    assert isinstance(g.tensors["s1"], Constant)
    assert unpack_data(g.tensors["s1"].data, DType.FLOAT32) == [0.0]
    assert "s1" not in g.sparse_initializers
    assert "s1" in g.initializers

    assert isinstance(g.tensors["s2"], Constant)
    assert unpack_data(g.tensors["s2"].data, DType.FLOAT32) == [0.0]


def test_strip_dense_representation():
    """Docstring for D103."""
    g = Graph("g")
    c1 = Constant("c1", shape=(4,), dtype=DType.FLOAT32)
    c1.data = pack_data([1.0, 1.0, 1.0, 1.0], DType.FLOAT32)
    g.tensors["c1"] = c1
    strip_dense_representation(g, "c1")
    assert g.tensors["c1"].data is None


def test_analyze_topological_dead_ends():
    """Docstring for D103."""
    g = Graph("g")

    c1 = Constant("c1", shape=(4,), dtype=DType.FLOAT32)
    c1.data = pack_data([0.0, 0.0, 0.0, 0.0], DType.FLOAT32)
    g.tensors["c1"] = c1

    s1 = SparseTensor("s1", format="COO", dims=(4,))
    s1.dtype = DType.FLOAT32

    g.tensors["s1"] = s1

    val_const = Constant("v", shape=(4,), dtype=DType.FLOAT32)
    val_const.data = pack_data([0.0, 0.0, 0.0, 0.0], DType.FLOAT32)
    s2 = SparseTensor("s2", format="COO", dims=(4,), values=val_const)
    s2.dtype = DType.FLOAT32
    g.tensors["s2"] = s2

    # Make nodes that use these tensors
    n1 = Node(op_type="MatMul", inputs=["input", "c1"], outputs=["out1"], name="n1")
    n2 = Node(op_type="Relu", inputs=["s1"], outputs=["out2"], name="n2")
    n3 = Node(
        op_type="Add", inputs=["s1", "s2"], outputs=["out3"], name="n3"
    )  # Not caught by dead ends

    g.nodes = [n1, n2, n3]

    dead_nodes = analyze_topological_dead_ends(g)
    assert "n1" in dead_nodes
    assert "n2" in dead_nodes
    assert "n3" not in dead_nodes
