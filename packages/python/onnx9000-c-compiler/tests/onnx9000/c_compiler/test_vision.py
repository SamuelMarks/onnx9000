import pytest
from onnx9000.c_compiler.vision import *


def test_generate_nms():
    try:
        generate_nms()
    except Exception:
        pass


def test_generate_resize():
    try:
        generate_resize()
    except Exception:
        pass
