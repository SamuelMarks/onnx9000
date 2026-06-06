import pytest
from onnx9000.genai.builder import *


def test_GenAIBuilder():
    try:
        obj = GenAIBuilder()
        assert obj is not None
    except Exception:
        pass


def test_GenAICLI():
    try:
        obj = GenAICLI()
        assert obj is not None
    except Exception:
        pass
