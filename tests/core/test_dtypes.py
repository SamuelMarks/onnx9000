import pytest
from onnx9000.core.dtypes import DType, to_cpp_type, to_emscripten_type


def test_to_cpp_type():
    assert to_cpp_type(DType.FLOAT32) == "float"
    assert to_cpp_type(DType.FLOAT64) == "double"
    assert to_cpp_type(DType.INT8) == "int8_t"
    assert to_cpp_type(DType.BOOL) == "bool"
    with pytest.raises(ValueError, match="mapped for DType"):
        to_cpp_type(DType.UNDEFINED)


def test_to_emscripten_type():
    assert to_emscripten_type(DType.FLOAT32) == "Float32Array"
    assert to_emscripten_type(DType.FLOAT64) == "Float64Array"
    assert to_emscripten_type(DType.INT64) == "BigInt64Array"
    assert to_emscripten_type(DType.UINT8) == "Uint8Array"
    with pytest.raises(ValueError, match="mapped for DType"):
        to_emscripten_type(DType.UNDEFINED)
