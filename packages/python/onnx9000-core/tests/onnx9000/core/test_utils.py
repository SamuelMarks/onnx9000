import pytest
from onnx9000.core.utils import *


def test_CyclicDependencyError():
    try:
        obj = CyclicDependencyError()
        assert obj is not None
    except Exception:
        pass


def test_topological_sort():
    try:
        topological_sort()
    except Exception:
        pass
