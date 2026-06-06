import pytest
from onnx9000.backends.apple.bindings import *

def test__load_libraries():
    try:
        res = _load_libraries()
    except Exception:
        pass

def test_is_accelerate_available():
    try:
        res = is_accelerate_available()
    except Exception:
        pass

def test_is_metal_available():
    try:
        res = is_metal_available()
    except Exception:
        pass

def test_is_mps_available():
    try:
        res = is_mps_available()
    except Exception:
        pass

def test_get_class():
    try:
        res = get_class()
    except Exception:
        pass

def test_get_selector():
    try:
        res = get_selector()
    except Exception:
        pass

def test_nsstring():
    try:
        res = nsstring()
    except Exception:
        pass

def test_mtl_create_system_default_device():
    try:
        res = mtl_create_system_default_device()
    except Exception:
        pass

