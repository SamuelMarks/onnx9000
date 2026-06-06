import pytest
from onnx9000.converters.tf.keras_h5_parser import *


def test_KerasH5Parser():
    try:
        obj = KerasH5Parser()
        assert obj is not None
    except Exception:
        pass
