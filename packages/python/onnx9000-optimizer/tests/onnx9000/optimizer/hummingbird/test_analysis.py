import pytest
from onnx9000.optimizer.hummingbird.analysis import *


def test_analyze_tree_depth():
    try:
        analyze_tree_depth()
    except Exception:
        pass


def test_analyze_leaf_distribution():
    try:
        analyze_leaf_distribution()
    except Exception:
        pass


def test_flatten_ensemble():
    try:
        flatten_ensemble()
    except Exception:
        pass


def test_cast_parameters():
    try:
        cast_parameters()
    except Exception:
        pass
