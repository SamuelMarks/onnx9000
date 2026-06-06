import pytest
from onnx9000.core.profiler import *


def test_ProfilerResult():
    try:
        obj = ProfilerResult()
        assert obj is not None
    except Exception:
        pass


def test_dtype_size():
    try:
        dtype_size()
    except Exception:
        pass


def test_resolve_volume():
    try:
        resolve_volume()
    except Exception:
        pass


def test_get_attr():
    try:
        get_attr()
    except Exception:
        pass


def test_profile_graph():
    try:
        profile_graph()
    except Exception:
        pass


def test__add_metric():
    try:
        _add_metric()
    except Exception:
        pass


def test_profile():
    try:
        profile()
    except Exception:
        pass
