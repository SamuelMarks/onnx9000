"""Unit tests for sparse tensor conversions and analysis in `onnx9000.core.sparse`."""

import os
import tempfile

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


def test_packing_and_sparse_format_conversions() -> None:
    """Verify pack/unpack, COO, CSR, CSC, BSR, and dense roundtrips."""
    assert get_struct_fmt(DType.FLOAT32) == "f"
    assert get_struct_fmt(DType.FLOAT16) == "e"
    assert get_byte_size(DType.FLOAT32) == 4

    raw = pack_data([1.0, 0.0, 0.0, 2.0], DType.FLOAT32)
    unpacked = unpack_data(raw, DType.FLOAT32)
    assert unpacked == [1.0, 0.0, 0.0, 2.0]

    packed_i8 = pack_sparse_int8([1, 2, 3], [0, 1, 2])
    assert len(packed_i8) == 5

    dense = Constant("w", values=raw, shape=(2, 2), dtype=DType.FLOAT32)
    coo = dense_to_coo(dense)
    csr = dense_to_csr(dense)
    csc = dense_to_csc(dense)
    bsr = dense_to_bsr(dense, (1, 1))

    assert coo.format == "COO"
    assert csr.format == "CSR"
    assert csc.format == "CSC"
    assert bsr.format == "BSR"

    for sp in [coo, csr, csc, bsr]:
        as_coo = sparse_to_coo(sp)
        assert as_coo.format == "COO"
        restored = sparse_to_dense(sp)
        assert unpack_data(restored.data, DType.FLOAT32) == [1.0, 0.0, 0.0, 2.0]


def test_sparsity_metrics_and_graph_passes() -> None:
    """Verify sparsity profiling, reports, MSE, de_sparsify, HF conversion, and dead-end analysis."""
    raw = pack_data([1.0, 0.0, 0.0, 0.0], DType.FLOAT32)
    dense = Constant("w", values=raw, shape=(2, 2), dtype=DType.FLOAT32)
    coo = dense_to_coo(dense)

    assert detect_theoretical_sparsity(dense) == 0.75
    assert calculate_memory_usage(dense) == 16
    assert calculate_memory_usage(coo) > 0
    assert isinstance(compression_ratio(dense, coo), float)
    assert evaluate_mse(dense, coo) == 0.0

    g = Graph("sparse_test")
    g.tensors["w"] = coo
    g.sparse_initializers.append("w")

    prof = profile(g)
    assert len(prof["layers"]) == 1
    rep = get_sparsity_report(g)
    assert "OVERALL" in rep

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        out_path = tf.name
    try:
        generate_json_report(g, out_path)
        assert os.path.exists(out_path)
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)

    map_sparse_to_safetensors(g)
    validate_provider_support(g, ["CPU", "LegacyEP"])

    nm_tensor = SparseTensor("nm_w", dims=(2, 2), format="NM")
    g.tensors["nm_w"] = nm_tensor
    resolve_nm_metadata(g)
    assert nm_tensor.metadata_props["sparse_encoding"] == "2:4"

    de_sparsify(g)
    assert isinstance(g.tensors["w"], Constant)

    hf_const = Constant("bitmask_w", values=raw, shape=(2, 2), dtype=DType.FLOAT32)
    hf_const.metadata_props = {"sparse_method": "prune"}
    g.tensors["bitmask_w"] = hf_const
    g.initializers.append("bitmask_w")
    convert_hf_sparse_to_onnx(g)
    assert isinstance(g.tensors["bitmask_w"], SparseTensor)

    zero_const = Constant(
        "zero_w", values=pack_data([0.0, 0.0], DType.FLOAT32), shape=(2,), dtype=DType.FLOAT32
    )
    g.tensors["zero_w"] = zero_const
    g.add_node(Node("MatMul", ["in_x", "zero_w"], ["dead_out"], name="dead_mm"))
    dead = analyze_topological_dead_ends(g)
    assert "dead_mm" in dead

    collapse_sparse_tensors(g)
    assert g.tensors["zero_w"].shape == (1,)

    strip_dense_representation(g, "zero_w")
    assert g.tensors["zero_w"].data is None
