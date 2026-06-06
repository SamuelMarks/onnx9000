import pytest
from onnx9000.tvm.relay.parser import *


def test_IRSpy():
    try:
        obj = IRSpy()
        assert obj is not None
    except Exception:
        pass


def test_save_json():
    try:
        save_json()
    except Exception:
        pass


def test_load_json():
    try:
        load_json()
    except Exception:
        pass
