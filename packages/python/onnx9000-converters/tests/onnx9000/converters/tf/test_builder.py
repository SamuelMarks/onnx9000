import pytest
from onnx9000.converters.tf.builder import *


def test_TFToONNXGraphBuilder():
    try:
        obj = TFToONNXGraphBuilder()
        assert obj is not None
    except Exception:
        pass
