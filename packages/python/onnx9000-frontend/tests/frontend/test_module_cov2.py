import pytest
from onnx9000.frontend.frontend.nn.module import Module


def test_module_register_buffer_invalid_type() -> None:
    m = Module()
    with pytest.raises(TypeError, match="cannot assign to buffer, must be a Tensor or None"):
        m.register_buffer("buf", "not_a_tensor")
