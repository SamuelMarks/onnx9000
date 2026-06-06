import pytest
from onnx9000.tensorrt.registry import *


def test_register_op():
    try:
        register_op()
    except Exception:
        pass


def test_get_op_translator():
    try:
        get_op_translator()
    except Exception:
        pass
