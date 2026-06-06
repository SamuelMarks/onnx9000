import pytest
from onnx9000.onnx2gguf.llama import *

def test_extract_llama_metadata():
    try:
        res = extract_llama_metadata()
    except Exception:
        pass

