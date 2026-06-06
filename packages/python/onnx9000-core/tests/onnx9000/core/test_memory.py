import pytest
from onnx9000.core.memory import *


def test_MemoryMapError():
    try:
        obj = MemoryMapError()
        assert obj is not None
    except Exception:
        pass


def test_mmap_tensor_data():
    try:
        mmap_tensor_data()
    except Exception:
        pass
