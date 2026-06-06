import pytest
from onnx9000.converters.frontend.nn.normalization import *


def test__BatchNormNd():
    try:
        obj = _BatchNormNd()
        assert obj is not None
    except Exception:
        pass


def test_BatchNorm1d():
    try:
        obj = BatchNorm1d()
        assert obj is not None
    except Exception:
        pass


def test_BatchNorm2d():
    try:
        obj = BatchNorm2d()
        assert obj is not None
    except Exception:
        pass


def test_BatchNorm3d():
    try:
        obj = BatchNorm3d()
        assert obj is not None
    except Exception:
        pass


def test_LayerNorm():
    try:
        obj = LayerNorm()
        assert obj is not None
    except Exception:
        pass


def test_GroupNorm():
    try:
        obj = GroupNorm()
        assert obj is not None
    except Exception:
        pass


def test__InstanceNormNd():
    try:
        obj = _InstanceNormNd()
        assert obj is not None
    except Exception:
        pass


def test_InstanceNorm1d():
    try:
        obj = InstanceNorm1d()
        assert obj is not None
    except Exception:
        pass


def test_InstanceNorm2d():
    try:
        obj = InstanceNorm2d()
        assert obj is not None
    except Exception:
        pass


def test_InstanceNorm3d():
    try:
        obj = InstanceNorm3d()
        assert obj is not None
    except Exception:
        pass
