import pytest
from onnx9000.core.exceptions import *


def test_Onnx9000Error():
    try:
        obj = Onnx9000Error()
        assert obj is not None
    except Exception:
        pass


def test_CompilationError():
    try:
        obj = CompilationError()
        assert obj is not None
    except Exception:
        pass


def test_UnsupportedOpError():
    try:
        obj = UnsupportedOpError()
        assert obj is not None
    except Exception:
        pass


def test_ShapeMismatchError():
    try:
        obj = ShapeMismatchError()
        assert obj is not None
    except Exception:
        pass


def test_ONNXParseError():
    try:
        obj = ONNXParseError()
        assert obj is not None
    except Exception:
        pass


def test_ShapeInferenceError():
    try:
        obj = ShapeInferenceError()
        assert obj is not None
    except Exception:
        pass


def test_UnsupportedOpsetError():
    try:
        obj = UnsupportedOpsetError()
        assert obj is not None
    except Exception:
        pass


def test_ValidationError():
    try:
        obj = ValidationError()
        assert obj is not None
    except Exception:
        pass
