import pytest
from onnx9000_skl2onnx import convert


def test_convert():
    with pytest.raises(ValueError):
        convert("")
    assert convert("test") == "[ONNX-IR] from skl test"
