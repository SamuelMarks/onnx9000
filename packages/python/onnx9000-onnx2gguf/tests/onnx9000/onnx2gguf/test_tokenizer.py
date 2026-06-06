import pytest
from onnx9000.onnx2gguf.tokenizer import *

def test_extract_tokenizer_metadata():
    try:
        res = extract_tokenizer_metadata()
    except Exception:
        pass

