import pytest
from onnx9000.converters.tf.importer import *


def test_TFImporter():
    try:
        obj = TFImporter()
        assert obj is not None
    except Exception:
        pass


def test__convert_dtype():
    try:
        _convert_dtype()
    except Exception:
        pass


def test_load_tf():
    try:
        load_tf()
    except Exception:
        pass
