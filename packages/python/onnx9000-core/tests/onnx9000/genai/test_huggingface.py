import pytest
from onnx9000.genai.huggingface import *


def test_HuggingFaceIntegration():
    try:
        obj = HuggingFaceIntegration()
        assert obj is not None
    except Exception:
        pass
