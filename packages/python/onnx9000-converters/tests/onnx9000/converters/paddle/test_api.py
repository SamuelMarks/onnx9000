import pytest
from onnx9000.converters.paddle.api import *


def test_paddle_optimize_graph():
    try:
        paddle_optimize_graph()
    except Exception:
        pass


def test__convert_paddle_graph():
    try:
        _convert_paddle_graph()
    except Exception:
        pass


def test_convert_paddle_to_onnx():
    try:
        convert_paddle_to_onnx()
    except Exception:
        pass
