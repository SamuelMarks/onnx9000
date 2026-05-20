import pytest
from onnx9000_keras2onnx import convert


def test_convert():
    assert convert("keras_model") == "[ONNX-IR] from keras keras_model"


def test_convert_invalid():
    with pytest.raises(ValueError):
        convert("")
