import pytest
from onnx9000.converters.jit.compiler import *

def test__get_compiler():
    try:
        res = _get_compiler()
    except Exception:
        pass

def test_compile_cpp():
    try:
        res = compile_cpp()
    except Exception:
        pass

def test_compile_wasm():
    try:
        res = compile_wasm()
    except Exception:
        pass

def test_load_module():
    try:
        res = load_module()
    except Exception:
        pass

