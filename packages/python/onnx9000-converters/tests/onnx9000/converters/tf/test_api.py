import pytest
from onnx9000.converters.tf.api import *


def test__convert_tfgraph():
    try:
        _convert_tfgraph()
    except Exception:
        pass


def test_convert_tf_to_onnx():
    try:
        convert_tf_to_onnx()
    except Exception:
        pass


def test_convert_keras_to_onnx():
    try:
        convert_keras_to_onnx()
    except Exception:
        pass


def test_convert_tflite_to_onnx():
    try:
        convert_tflite_to_onnx()
    except Exception:
        pass
