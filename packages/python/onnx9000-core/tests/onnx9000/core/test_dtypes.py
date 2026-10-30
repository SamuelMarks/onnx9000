"""Unit tests for DType mappings in `onnx9000.core.dtypes`."""

import pytest
from onnx9000.core.dtypes import DType, to_cpp_type, to_emscripten_type


def test_dtypes_mappings() -> None:
    """Verify DType enum, to_cpp_type, and to_emscripten_type."""
    assert DType.FLOAT32.value == 1
    assert to_cpp_type(DType.FLOAT32) == "float"
    assert to_emscripten_type(DType.FLOAT32) == "Float32Array"

    with pytest.raises(ValueError):
        to_cpp_type(DType.UNDEFINED)
    with pytest.raises(ValueError):
        to_emscripten_type(DType.UNDEFINED)
