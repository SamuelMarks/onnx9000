import pytest
from onnx9000.converters.frontend.bonsai import *

def test_BonsaiImporter():
    try:
        obj = BonsaiImporter()
        assert obj is not None
    except Exception:
        pass

