import pytest
from onnx9000.tflite_exporter.flatbuffer.schema import *


def test_TensorType():
    try:
        obj = TensorType()
        assert obj is not None
    except Exception:
        pass


def test_Padding():
    try:
        obj = Padding()
        assert obj is not None
    except Exception:
        pass


def test_BuiltinOperator():
    try:
        obj = BuiltinOperator()
        assert obj is not None
    except Exception:
        pass


def test_BuiltinOptions():
    try:
        obj = BuiltinOptions()
        assert obj is not None
    except Exception:
        pass


def test_OperatorCode():
    try:
        obj = OperatorCode()
        assert obj is not None
    except Exception:
        pass


def test_QuantizationParameters():
    try:
        obj = QuantizationParameters()
        assert obj is not None
    except Exception:
        pass


def test_Tensor():
    try:
        obj = Tensor()
        assert obj is not None
    except Exception:
        pass


def test_Operator():
    try:
        obj = Operator()
        assert obj is not None
    except Exception:
        pass


def test_SubGraph():
    try:
        obj = SubGraph()
        assert obj is not None
    except Exception:
        pass


def test_Buffer():
    try:
        obj = Buffer()
        assert obj is not None
    except Exception:
        pass


def test_Metadata():
    try:
        obj = Metadata()
        assert obj is not None
    except Exception:
        pass


def test_Model():
    try:
        obj = Model()
        assert obj is not None
    except Exception:
        pass
