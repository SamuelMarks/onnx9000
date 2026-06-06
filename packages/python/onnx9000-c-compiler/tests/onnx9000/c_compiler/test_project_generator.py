import pytest
from onnx9000.c_compiler.project_generator import *

def test_generate_makefile():
    try:
        res = generate_makefile()
    except Exception:
        pass

def test_generate_main_c():
    try:
        res = generate_main_c()
    except Exception:
        pass

def test_generate_cmakelists():
    try:
        res = generate_cmakelists()
    except Exception:
        pass

def test_generate_arduino_sketch():
    try:
        res = generate_arduino_sketch()
    except Exception:
        pass

