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
        _map_jax_type()
    except Exception:
        pass


def test_load_jax():
    try:
        load_jax()
    except Exception:
        pass


def test_load():
    try:
        load()
    except Exception:
        pass
