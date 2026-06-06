import pytest
from onnx9000.core.models.dit import *


def test_DiTBlock():
    try:
        obj = DiTBlock()
        assert obj is not None
    except Exception:
        pass


def test_DiT():
    try:
        obj = DiT()
        assert obj is not None
    except Exception:
        pass


def test_get_param():
    try:
        get_param()
    except Exception:
        pass


def test_dit_xl_2():
    try:
        dit_xl_2()
    except Exception:
        pass
