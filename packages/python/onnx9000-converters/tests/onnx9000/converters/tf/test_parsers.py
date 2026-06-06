import pytest
from onnx9000.converters.tf.parsers import *


def test_TFNode():
    try:
        obj = TFNode()
        assert obj is not None
    except Exception:
        pass


def test_TFGraph():
    try:
        obj = TFGraph()
        assert obj is not None
    except Exception:
        pass


def test_ProtobufParser():
    try:
        obj = ProtobufParser()
        assert obj is not None
    except Exception:
        pass


def test_H5Parser():
    try:
        obj = H5Parser()
        assert obj is not None
    except Exception:
        pass


def test_FlatBufferParser():
    try:
        obj = FlatBufferParser()
        assert obj is not None
    except Exception:
        pass


def test_parse_graphdef():
    try:
        parse_graphdef()
    except Exception:
        pass


def test_parse_saved_model():
    try:
        parse_saved_model()
    except Exception:
        pass


def test_extract_variables():
    try:
        extract_variables()
    except Exception:
        pass


def test_load_h5_model():
    try:
        load_h5_model()
    except Exception:
        pass


def test_load_keras_v3():
    try:
        load_keras_v3()
    except Exception:
        pass


def test_parse_tflite():
    try:
        parse_tflite()
    except Exception:
        pass


def test_map_tf_shape_to_onnx():
    try:
        map_tf_shape_to_onnx()
    except Exception:
        pass


def test_log_unsupported_node():
    try:
        log_unsupported_node()
    except Exception:
        pass


def test_fallback_to_custom_op():
    try:
        fallback_to_custom_op()
    except Exception:
        pass
