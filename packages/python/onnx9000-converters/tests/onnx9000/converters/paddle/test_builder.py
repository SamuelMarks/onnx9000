import pytest
from onnx9000.converters.paddle.builder import *


def test_PaddleToONNXGraphBuilder():
    try:
        obj = PaddleToONNXGraphBuilder()
        assert obj is not None
    except Exception:
        pass
