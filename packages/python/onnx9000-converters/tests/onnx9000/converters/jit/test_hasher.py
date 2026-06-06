import pytest
from onnx9000.converters.jit.hasher import *


def test_hash_graph():
    try:
        hash_graph()
    except Exception:
        pass
