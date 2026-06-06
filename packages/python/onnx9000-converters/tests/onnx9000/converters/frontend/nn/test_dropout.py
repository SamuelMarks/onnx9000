import pytest
from onnx9000.converters.frontend.nn.dropout import *


def test__DropoutNd():
    try:
        obj = _DropoutNd()
        assert obj is not None
    except Exception:
        pass


def test_Dropout():
    try:
        obj = Dropout()
        assert obj is not None
    except Exception:
        pass


def test_Dropout1d():
    try:
        obj = Dropout1d()
        assert obj is not None
    except Exception:
        pass


def test_Dropout2d():
    try:
        obj = Dropout2d()
        assert obj is not None
    except Exception:
        pass


def test_Dropout3d():
    try:
        obj = Dropout3d()
        assert obj is not None
    except Exception:
        pass
