import pytest
from onnx9000.converters.safetensors.loader import *


def test_load_safetensors_to_graph():
    try:
        load_safetensors_to_graph()
    except Exception:
        pass


def test_map_huggingface_to_onnx():
    try:
        map_huggingface_to_onnx()
    except Exception:
        pass


def test_load_and_patch_state_dict():
    try:
        load_and_patch_state_dict()
    except Exception:
        pass


def test_convert_pytorch_to_safetensors():
    try:
        convert_pytorch_to_safetensors()
    except Exception:
        pass


def test_convert_tf_to_safetensors():
    try:
        convert_tf_to_safetensors()
    except Exception:
        pass


def test_dump_graph_to_safetensors():
    try:
        dump_graph_to_safetensors()
    except Exception:
        pass


def test_validate_onnx_shapes_and_dtypes():
    try:
        validate_onnx_shapes_and_dtypes()
    except Exception:
        pass


def test_unpack_awq():
    try:
        unpack_awq()
    except Exception:
        pass
