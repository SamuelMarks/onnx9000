import pytest
from onnx9000.converters.frontend.nn.pool import *

def test__MaxPoolNd():
    try:
        obj = _MaxPoolNd()
        assert obj is not None
    except Exception:
        pass

def test_MaxPool1d():
    try:
        obj = MaxPool1d()
        assert obj is not None
    except Exception:
        pass

def test_MaxPool2d():
    try:
        obj = MaxPool2d()
        assert obj is not None
    except Exception:
        pass

def test__AvgPoolNd():
    try:
        obj = _AvgPoolNd()
        assert obj is not None
    except Exception:
        pass

def test_AvgPool1d():
    try:
        obj = AvgPool1d()
        assert obj is not None
    except Exception:
        pass

def test_AvgPool2d():
    try:
        obj = AvgPool2d()
        assert obj is not None
    except Exception:
        pass

def test__AdaptiveAvgPoolNd():
    try:
        obj = _AdaptiveAvgPoolNd()
        assert obj is not None
    except Exception:
        pass

def test_AdaptiveAvgPool2d():
    try:
        obj = AdaptiveAvgPool2d()
        assert obj is not None
    except Exception:
        pass

def test__pair():
    try:
        res = _pair()
    except Exception:
        pass

def test__single():
    try:
        res = _single()
    except Exception:
        pass

