import pytest
from onnx9000.core.config import *

def test_module_load():
    import onnx9000.core.config
    assert onnx9000.core.config is not None

