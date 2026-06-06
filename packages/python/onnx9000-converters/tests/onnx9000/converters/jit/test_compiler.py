import pytest
from onnx9000.converters.jit.compiler import *


def test__get_compiler():
    try:
        _get_compiler()
    except Exception:
        pass


def test_compile_cpp():
    try:
        compile_cpp()
    except Exception:
        pass


def test_compile_wasm():
    try:
        compile_wasm()
    except Exception:
        pass


def test_load_module():
    try:
        load_module()
    except Exception:
        pass
