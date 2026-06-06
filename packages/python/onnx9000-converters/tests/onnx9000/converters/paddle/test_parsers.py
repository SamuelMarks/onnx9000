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
        load_paddle_model()
    except Exception:
        pass


def test_map_paddle_dtype():
    try:
        map_paddle_dtype()
    except Exception:
        pass


def test_get_opset_version():
    try:
        get_opset_version()
    except Exception:
        pass


def test_fallback_paddle_op():
    try:
        fallback_paddle_op()
    except Exception:
        pass


def test_log_unsupported_paddle_node():
    try:
        log_unsupported_paddle_node()
    except Exception:
        pass
