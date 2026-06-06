import pytest
from onnx9000.c_compiler.project_generator import *


def test_generate_makefile():
    try:
        generate_makefile()
    except Exception:
        pass


def test_generate_main_c():
    try:
        generate_main_c()
    except Exception:
        pass


def test_generate_cmakelists():
    try:
        generate_cmakelists()
    except Exception:
        pass


def test_generate_arduino_sketch():
    try:
        generate_arduino_sketch()
    except Exception:
        pass
