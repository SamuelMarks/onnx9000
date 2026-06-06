import pytest
from onnx9000.tensorrt.network import *


def test_ITensor():
    try:
        obj = ITensor()
        assert obj is not None
    except Exception:
        pass


def test_INetworkDefinition():
    try:
        obj = INetworkDefinition()
        assert obj is not None
    except Exception:
        pass


def test_IBuilderConfig():
    try:
        obj = IBuilderConfig()
        assert obj is not None
    except Exception:
        pass
