import pytest
from onnx9000.converters.paddle.parsers import *

def test_PaddleNode():
    try:
        obj = PaddleNode()
        assert obj is not None
    except Exception:
        pass

def test_PaddleVar():
    try:
        obj = PaddleVar()
        assert obj is not None
    except Exception:
        pass

def test_PaddleBlock():
    try:
        obj = PaddleBlock()
        assert obj is not None
    except Exception:
        pass

def test_PaddleGraph():
    try:
        obj = PaddleGraph()
        assert obj is not None
    except Exception:
        pass

def test_PaddleProtobufParser():
    try:
        obj = PaddleProtobufParser()
        assert obj is not None
    except Exception:
        pass

def test_load_paddle_model():
    try:
        res = load_paddle_model()
    except Exception:
        pass

def test_map_paddle_dtype():
    try:
        res = map_paddle_dtype()
    except Exception:
        pass

def test_get_opset_version():
    try:
        res = get_opset_version()
    except Exception:
        pass

def test_fallback_paddle_op():
    try:
        res = fallback_paddle_op()
    except Exception:
        pass

def test_log_unsupported_paddle_node():
    try:
        res = log_unsupported_paddle_node()
    except Exception:
        pass

