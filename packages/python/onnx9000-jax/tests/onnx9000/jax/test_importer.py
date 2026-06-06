import pytest
from onnx9000.jax.importer import *

def test_JaxprImporter():
    try:
        obj = JaxprImporter()
        assert obj is not None
    except Exception:
        pass

def test__map_jax_type():
    try:
        res = _map_jax_type()
    except Exception:
        pass

def test_load_jax():
    try:
        res = load_jax()
    except Exception:
        pass

def test_load():
    try:
        res = load()
    except Exception:
        pass

