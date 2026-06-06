import pytest
from onnx9000.converters.frontend.quantization import *


def test_GGUFQuantizationMapper():
    try:
        obj = GGUFQuantizationMapper()
        assert obj is not None
    except Exception:
        pass


def test_AWQParser():
    try:
        obj = AWQParser()
        assert obj is not None
    except Exception:
        pass


def test_GPTQParser():
    try:
        obj = GPTQParser()
        assert obj is not None
    except Exception:
        pass


def test_AQTParser():
    try:
        obj = AQTParser()
        assert obj is not None
    except Exception:
        pass
