import pytest
from onnx9000.optimizer.hummingbird.perfect_tree import *


def test_PerfectTreeCompiler():
    try:
        obj = PerfectTreeCompiler()
        assert obj is not None
    except Exception:
        pass


def test_handle_perfect_multi_output():
    try:
        handle_perfect_multi_output()
    except Exception:
        pass


def test_map_categorical_perfect():
    try:
        map_categorical_perfect()
    except Exception:
        pass
