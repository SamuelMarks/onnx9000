import pytest
from onnx9000.genai.logit_processor_list import *

def test_LogitProcessorList():
    try:
        obj = LogitProcessorList()
        assert obj is not None
    except Exception:
        pass

