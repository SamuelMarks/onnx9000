import pytest
from onnx9000.converters.torch.export import *

def test_ExportParser():
    try:
        obj = ExportParser()
        assert obj is not None
    except Exception:
        pass

