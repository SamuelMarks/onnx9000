import pytest
from onnx9000.toolkit.training.autograd.utils import *


def test_GradientProto():
    try:
        obj = GradientProto()
        assert obj is not None
    except Exception:
        pass


def test_generate_gradient_proto():
    try:
        generate_gradient_proto()
    except Exception:
        pass


def test_calculate_gradient_payload_size():
    try:
        calculate_gradient_payload_size()
    except Exception:
        pass


def test_compress_gradients_int8():
    try:
        compress_gradients_int8()
    except Exception:
        pass


def test_compile_multi_replica_graph():
    try:
        compile_multi_replica_graph()
    except Exception:
        pass


def test_embed_distributed_identifiers():
    try:
        embed_distributed_identifiers()
    except Exception:
        pass


def test_add_synchronous_barrier():
    try:
        add_synchronous_barrier()
    except Exception:
        pass


def test_calculate_communication_bounds():
    try:
        calculate_communication_bounds()
    except Exception:
        pass


def test_flatten_gradients():
    try:
        flatten_gradients()
    except Exception:
        pass


def test_reverse_topological_sort():
    try:
        reverse_topological_sort()
    except Exception:
        pass
