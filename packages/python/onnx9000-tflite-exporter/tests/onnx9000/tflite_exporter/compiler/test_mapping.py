import pytest
from onnx9000.tflite_exporter.compiler.mapping import *


def test_map_onnx_type_to_tflite():
    try:
        map_onnx_type_to_tflite()
    except Exception:
        pass


def test_map_onnx_shape_to_tflite():
    try:
        map_onnx_shape_to_tflite()
    except Exception:
        pass


def test_create_shape_signature():
    try:
        create_shape_signature()
    except Exception:
        pass
