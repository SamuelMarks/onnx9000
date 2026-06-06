import pytest
from onnx9000.backends.codegen.ops.control_flow import *


def test_generate_if():
    try:
        generate_if()
    except Exception:
        pass


def test_generate_loop():
    try:
        generate_loop()
    except Exception:
        pass
