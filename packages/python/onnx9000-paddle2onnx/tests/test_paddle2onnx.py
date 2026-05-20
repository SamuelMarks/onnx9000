import pytest
from onnx9000_paddle2onnx import convert


def test_convert():
    assert convert("paddle_model") == "[ONNX-IR] from paddle_model"


def test_convert_invalid():
    with pytest.raises(ValueError):
        convert("")
