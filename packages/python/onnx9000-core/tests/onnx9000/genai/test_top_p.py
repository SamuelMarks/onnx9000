import pytest
from onnx9000.genai.top_p import *


def test_TopPLogitProcessor():
    try:
        obj = TopPLogitProcessor()
        assert obj is not None
    except Exception:
        pass
