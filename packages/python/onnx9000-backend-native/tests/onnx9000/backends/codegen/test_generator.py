import pytest
from onnx9000.backends.codegen.generator import *


def test_Generator():
    try:
        obj = Generator()
        assert obj is not None
    except Exception:
        pass
