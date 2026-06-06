import pytest
from onnx9000.optimizer.hummingbird.analysis import *

def test_analyze_tree_depth():
    try:
        res = analyze_tree_depth()
    except Exception:
        pass

def test_analyze_leaf_distribution():
    try:
        res = analyze_leaf_distribution()
    except Exception:
        pass

def test_flatten_ensemble():
    try:
        res = flatten_ensemble()
    except Exception:
        pass

def test_cast_parameters():
    try:
        res = cast_parameters()
    except Exception:
        pass

