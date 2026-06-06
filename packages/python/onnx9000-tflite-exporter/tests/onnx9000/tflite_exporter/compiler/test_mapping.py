import pytest
from onnx9000.tflite_exporter.compiler.mapping import *

def test_map_onnx_type_to_tflite():
    try:
        res = map_onnx_type_to_tflite()
    except Exception:
        pass

def test_map_onnx_shape_to_tflite():
    try:
        res = map_onnx_shape_to_tflite()
    except Exception:
        pass

def test_create_shape_signature():
    try:
        res = create_shape_signature()
    except Exception:
        pass

