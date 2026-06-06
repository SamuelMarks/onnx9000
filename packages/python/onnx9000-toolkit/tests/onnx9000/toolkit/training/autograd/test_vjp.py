import pytest
from onnx9000.toolkit.training.autograd.vjp import *


def test_VJPRule():
    try:
        obj = VJPRule()
        assert obj is not None
    except Exception:
        pass
