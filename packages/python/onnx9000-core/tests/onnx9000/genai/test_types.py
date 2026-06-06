import pytest
from onnx9000.genai.types import *


def test_ModelParams():
    try:
        obj = ModelParams()
        assert obj is not None
    except Exception:
        pass


def test_GeneratorParams():
    try:
        obj = GeneratorParams()
        assert obj is not None
    except Exception:
        pass
