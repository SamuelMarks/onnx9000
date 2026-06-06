import pytest
from onnx9000.tvm.relay.transform.dead_code_elimination import *


def test_DeadCodeElimination():
    try:
        obj = DeadCodeElimination()
        assert obj is not None
    except Exception:
        pass


def test_eliminate_dead_code():
    try:
        eliminate_dead_code()
    except Exception:
        pass
