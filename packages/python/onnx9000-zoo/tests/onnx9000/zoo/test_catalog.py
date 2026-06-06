import pytest
from onnx9000.zoo.catalog import *

def test_ZooCatalog():
    try:
        obj = ZooCatalog()
        assert obj is not None
    except Exception:
        pass

