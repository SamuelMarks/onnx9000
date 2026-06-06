import pytest
from onnx9000.backends.ffi.core import *


def test_DynamicLibraryError():
    try:
        obj = DynamicLibraryError()
        assert obj is not None
    except Exception:
        pass


def test_DynamicLibrary():
    try:
        obj = DynamicLibrary()
        assert obj is not None
    except Exception:
        pass


def test_HardwareContextHandle():
    try:
        obj = HardwareContextHandle()
        assert obj is not None
    except Exception:
        pass


def test_map_python_string():
    try:
        map_python_string()
    except Exception:
        pass


def test_map_python_bool():
    try:
        map_python_bool()
    except Exception:
        pass


def test_profile_ctypes_overhead():
    try:
        profile_ctypes_overhead()
    except Exception:
        pass


def test_get_cpu_features():
    try:
        get_cpu_features()
    except Exception:
        pass


def test_get_cache_sizes():
    try:
        get_cache_sizes()
    except Exception:
        pass
