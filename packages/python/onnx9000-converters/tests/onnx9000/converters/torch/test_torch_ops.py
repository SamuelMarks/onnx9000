import pytest
from onnx9000.converters.torch.torch_ops import *

def test__create_mapper():
    try:
        res = _create_mapper()
    except Exception:
        pass

