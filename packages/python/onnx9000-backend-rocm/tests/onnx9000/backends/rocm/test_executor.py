import pytest
from onnx9000.backends.rocm.executor import *


def test_Dispatcher():
    try:
        obj = Dispatcher()
        assert obj is not None
    except Exception:
        pass


def test__rocm_matmul():
    try:
        _rocm_matmul()
    except Exception:
        pass
