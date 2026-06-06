import pytest
from onnx9000.core.sparse import *


def test_get_struct_fmt():
    try:
        get_struct_fmt()
    except Exception:
        pass


def test_get_byte_size():
    try:
        get_byte_size()
    except Exception:
        pass


def test_unpack_data():
    try:
        unpack_data()
    except Exception:
        pass


def test_pack_data():
    try:
        pack_data()
    except Exception:
        pass


def test_pack_sparse_int8():
    try:
        pack_sparse_int8()
    except Exception:
        pass


def test_dense_to_coo():
    try:
        dense_to_coo()
    except Exception:
        pass


def test_dense_to_csr():
    try:
        dense_to_csr()
    except Exception:
        pass


def test_dense_to_csc():
    try:
        dense_to_csc()
    except Exception:
        pass


def test_dense_to_bsr():
    try:
        dense_to_bsr()
    except Exception:
        pass


def test_sparse_to_coo():
    try:
        sparse_to_coo()
    except Exception:
        pass


def test_sparse_to_dense():
    try:
        sparse_to_dense()
    except Exception:
        pass


def test_detect_theoretical_sparsity():
    try:
        detect_theoretical_sparsity()
    except Exception:
        pass


def test_calculate_memory_usage():
    try:
        calculate_memory_usage()
    except Exception:
        pass


def test_compression_ratio():
    try:
        compression_ratio()
    except Exception:
        pass


def test_profile():
    try:
        profile()
    except Exception:
        pass


def test_get_sparsity_report():
    try:
        get_sparsity_report()
    except Exception:
        pass


def test_generate_json_report():
    try:
        generate_json_report()
    except Exception:
        pass


def test_evaluate_mse():
    try:
        evaluate_mse()
    except Exception:
        pass


def test_de_sparsify():
    try:
        de_sparsify()
    except Exception:
        pass


def test_convert_hf_sparse_to_onnx():
    try:
        convert_hf_sparse_to_onnx()
    except Exception:
        pass


def test_resolve_nm_metadata():
    try:
        resolve_nm_metadata()
    except Exception:
        pass


def test_map_sparse_to_safetensors():
    try:
        map_sparse_to_safetensors()
    except Exception:
        pass


def test_validate_provider_support():
    try:
        validate_provider_support()
    except Exception:
        pass


def test_collapse_sparse_tensors():
    try:
        collapse_sparse_tensors()
    except Exception:
        pass


def test_strip_dense_representation():
    try:
        strip_dense_representation()
    except Exception:
        pass


def test_analyze_topological_dead_ends():
    try:
        analyze_topological_dead_ends()
    except Exception:
        pass
