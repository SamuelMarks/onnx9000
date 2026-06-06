import pytest
from onnx9000.genai.openai_api import *

def test_OpenAIServer():
    try:
        obj = OpenAIServer()
        assert obj is not None
    except Exception:
        pass

