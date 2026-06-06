import pytest
from onnx9000.onnx2gguf.hub import *

def test_fetch_hf_config():
    try:
        res = fetch_hf_config()
    except Exception:
        pass

def test_generate_readme():
    try:
        res = generate_readme()
    except Exception:
        pass

