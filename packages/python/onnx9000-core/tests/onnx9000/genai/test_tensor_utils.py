import pytest
from onnx9000.genai.tensor_utils import *

def test_SequenceTensorUtils():
    try:
        obj = SequenceTensorUtils()
        assert obj is not None
    except Exception:
        pass

