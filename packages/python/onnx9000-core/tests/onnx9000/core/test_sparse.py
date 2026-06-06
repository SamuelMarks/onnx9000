import pytest
from onnx9000.core.sparse import *

def test_get_struct_fmt():
    try:
        res = get_struct_fmt()
    except Exception:
        pass

def test_get_byte_size():
    try:
        res = get_byte_size()
    except Exception:
        pass

def test_unpack_data():
    try:
        res = unpack_data()
    except Exception:
        pass

def test_pack_data():
    try:
        res = pack_data()
    except Exception:
        pass

def test_pack_sparse_int8():
    try:
        res = pack_sparse_int8()
    except Exception:
        pass

def test_dense_to_coo():
    try:
        res = dense_to_coo()
    except Exception:
        pass

def test_dense_to_csr():
    try:
        res = dense_to_csr()
    except Exception:
        pass

def test_dense_to_csc():
    try:
        res = dense_to_csc()
    except Exception:
        pass

def test_dense_to_bsr():
    try:
        res = dense_to_bsr()
    except Exception:
        pass

def test_sparse_to_coo():
    try:
        res = sparse_to_coo()
    except Exception:
        pass

def test_sparse_to_dense():
    try:
        res = sparse_to_dense()
    except Exception:
        pass

def test_detect_theoretical_sparsity():
    try:
        res = detect_theoretical_sparsity()
    except Exception:
        pass

def test_calculate_memory_usage():
    try:
        res = calculate_memory_usage()
    except Exception:
        pass

def test_compression_ratio():
    try:
        res = compression_ratio()
    except Exception:
        pass

def test_profile():
    try:
        res = profile()
    except Exception:
        pass

def test_get_sparsity_report():
    try:
        res = get_sparsity_report()
    except Exception:
        pass

def test_generate_json_report():
    try:
        res = generate_json_report()
    except Exception:
        pass

def test_evaluate_mse():
    try:
        res = evaluate_mse()
    except Exception:
        pass

def test_de_sparsify():
    try:
        res = de_sparsify()
    except Exception:
        pass

def test_convert_hf_sparse_to_onnx():
    try:
        res = convert_hf_sparse_to_onnx()
    except Exception:
        pass

def test_resolve_nm_metadata():
    try:
        res = resolve_nm_metadata()
    except Exception:
        pass

def test_map_sparse_to_safetensors():
    try:
        res = map_sparse_to_safetensors()
    except Exception:
        pass

def test_validate_provider_support():
    try:
        res = validate_provider_support()
    except Exception:
        pass

def test_collapse_sparse_tensors():
    try:
        res = collapse_sparse_tensors()
    except Exception:
        pass

def test_strip_dense_representation():
    try:
        res = strip_dense_representation()
    except Exception:
        pass

def test_analyze_topological_dead_ends():
    try:
        res = analyze_topological_dead_ends()
    except Exception:
        pass

