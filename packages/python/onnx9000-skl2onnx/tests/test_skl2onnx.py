import pytest
from onnx9000_skl2onnx import convert


def test_convert():
    assert convert("skl_model") == "[ONNX-IR] from skl skl_model"


def test_convert_invalid():
    with pytest.raises(ValueError):
        convert("")
