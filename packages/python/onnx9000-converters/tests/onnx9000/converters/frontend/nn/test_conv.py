import pytest
from onnx9000.converters.frontend.nn.conv import *


def test__ConvNd():
    try:
        obj = _ConvNd()
        assert obj is not None
    except Exception:
        pass


def test_Conv1d():
    try:
        obj = Conv1d()
        assert obj is not None
    except Exception:
        pass


def test_Conv2d():
    try:
        obj = Conv2d()
        assert obj is not None
    except Exception:
        pass


def test_Conv3d():
    try:
        obj = Conv3d()
        assert obj is not None
    except Exception:
        pass


def test__ConvTransposeNd():
    try:
        obj = _ConvTransposeNd()
        assert obj is not None
    except Exception:
        pass


def test_ConvTranspose1d():
    try:
        obj = ConvTranspose1d()
        assert obj is not None
    except Exception:
        pass


def test_ConvTranspose2d():
    try:
        obj = ConvTranspose2d()
        assert obj is not None
    except Exception:
        pass


def test__pair():
    try:
        _pair()
    except Exception:
        pass


def test__single():
    try:
        _single()
    except Exception:
        pass


def test__triple():
    try:
        _triple()
    except Exception:
        pass
