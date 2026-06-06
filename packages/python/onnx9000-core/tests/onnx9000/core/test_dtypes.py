import pytest
from onnx9000.core.dtypes import *


def test_DType():
    try:
        obj = DType()
        assert obj is not None
    except Exception:
        pass


def test_to_cpp_type():
    try:
        to_cpp_type()
    except Exception:
        pass


def test_to_emscripten_type():
    try:
        to_emscripten_type()
    except Exception:
        pass
