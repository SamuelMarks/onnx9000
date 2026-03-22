import pytest
from onnx9000.tflite_exporter.compiler.mapping import (
    map_onnx_type_to_tflite,
    map_onnx_shape_to_tflite,
    create_shape_signature,
)
from onnx9000.tflite_exporter.flatbuffer.schema import TensorType


def test_map_onnx_type_to_tflite():
    assert map_onnx_type_to_tflite("float32") == TensorType.FLOAT32
    assert map_onnx_type_to_tflite("float16") == TensorType.FLOAT16
    assert map_onnx_type_to_tflite("int32") == TensorType.INT32
    assert map_onnx_type_to_tflite("int64") == TensorType.INT32
    assert map_onnx_type_to_tflite("int64", name="test") == TensorType.INT32
    assert map_onnx_type_to_tflite("int8") == TensorType.INT8
    assert map_onnx_type_to_tflite("uint8") == TensorType.UINT8
    assert map_onnx_type_to_tflite("bool") == TensorType.BOOL
    assert map_onnx_type_to_tflite("string") == TensorType.STRING
    assert map_onnx_type_to_tflite("float64") == TensorType.FLOAT32
    assert map_onnx_type_to_tflite("unknown") == TensorType.FLOAT32


def test_map_onnx_shape_to_tflite():
    assert map_onnx_shape_to_tflite(None) == []
    assert map_onnx_shape_to_tflite([]) == []
    assert map_onnx_shape_to_tflite([1, 2, 3]) == [1, 2, 3]
    assert map_onnx_shape_to_tflite([-1, 2, 3]) == [-1, 2, 3]
    assert map_onnx_shape_to_tflite(["dynamic", 2, 3]) == [-1, 2, 3]


def test_create_shape_signature():
    assert create_shape_signature(["dynamic", 2, 3]) == [-1, 2, 3]
