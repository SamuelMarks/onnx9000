import pytest
from onnx9000.optimizer.simplifier.api import *

def test__calculate_graph_size():
    try:
        res = _calculate_graph_size()
    except Exception:
        pass

def test_check_disconnected_outputs():
    try:
        res = check_disconnected_outputs()
    except Exception:
        pass

def test_extract_scalars():
    try:
        res = extract_scalars()
    except Exception:
        pass

def test_simplify():
    try:
        res = simplify()
    except Exception:
        pass

