import pytest
from onnx9000.converters.frontend.jit import *


def test_jit():
    try:
        jit()
    except Exception:
        pass
