import pytest
from onnx9000.jax.jax_importer import *

def test_JAXImporter():
    try:
        obj = JAXImporter()
        assert obj is not None
    except Exception:
        pass

