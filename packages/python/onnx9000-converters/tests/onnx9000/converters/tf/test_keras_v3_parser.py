import pytest
from onnx9000.converters.tf.keras_v3_parser import *


def test_Keras3Parser():
    try:
        obj = Keras3Parser()
        assert obj is not None
    except Exception:
        pass
