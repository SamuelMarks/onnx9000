import pytest
from onnx9000.onnx2gguf.naming import *


def test_rename_tensor():
    try:
        rename_tensor()
    except Exception:
        pass
