import pytest
from onnx9000.backends.session import InferenceSession


def test_typeerror():
    with pytest.raises(TypeError, match="InferenceSession requires an IR Graph object"):
        InferenceSession("invalid")
