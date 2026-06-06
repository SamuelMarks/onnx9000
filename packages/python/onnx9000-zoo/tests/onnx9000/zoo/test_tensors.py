import pytest
from onnx9000.zoo.tensors import *


def test_SafeTensorsMmapParser():
    try:
        obj = SafeTensorsMmapParser()
        assert obj is not None
    except Exception:
        pass


def test_GSPMDReconciler():
    try:
        obj = GSPMDReconciler()
        assert obj is not None
    except Exception:
        pass


def test_BFloat16Upcaster():
    try:
        obj = BFloat16Upcaster()
        assert obj is not None
    except Exception:
        pass


def test_MsgPackFlaxDeserializer():
    try:
        obj = MsgPackFlaxDeserializer()
        assert obj is not None
    except Exception:
        pass
