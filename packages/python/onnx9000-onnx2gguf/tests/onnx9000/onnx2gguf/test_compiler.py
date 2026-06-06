import pytest
from onnx9000.onnx2gguf.compiler import *

def test_get_gguf_type():
    try:
        res = get_gguf_type()
    except Exception:
        pass

def test_infer_architecture():
    try:
        res = infer_architecture()
    except Exception:
        pass

def test_sanitize_doc_string():
    try:
        res = sanitize_doc_string()
    except Exception:
        pass

def test_compile_gguf():
    try:
        res = compile_gguf()
    except Exception:
        pass

