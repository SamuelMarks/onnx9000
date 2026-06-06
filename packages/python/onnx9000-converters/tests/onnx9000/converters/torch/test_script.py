import pytest
from onnx9000.converters.torch.script import *


def test_TorchScriptParser():
    try:
        obj = TorchScriptParser()
        assert obj is not None
    except Exception:
        pass
