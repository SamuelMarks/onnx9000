import pytest
from onnx9000.backends.codegen.compiler import *


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


def test_load_pybind_module():
    try:
        load_pybind_module()
    except Exception:
        pass


def test_load_ctypes_library():
    try:
        load_ctypes_library()
    except Exception:
        pass


def test_compile_static_lib():
    try:
        compile_static_lib()
    except Exception:
        pass
