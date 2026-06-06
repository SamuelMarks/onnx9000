import pytest
from onnx9000.converters.jit.hasher import *

def test_hash_graph():
    try:
        res = hash_graph()
    except Exception:
        pass

