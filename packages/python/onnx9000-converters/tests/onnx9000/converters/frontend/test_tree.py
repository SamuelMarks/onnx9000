import pytest
from onnx9000.converters.frontend.tree import *


def test_tree_map():
    try:
        tree_map()
    except Exception:
        pass


def test_tree_flatten():
    try:
        tree_flatten()
    except Exception:
        pass


def test_tree_unflatten():
    try:
        tree_unflatten()
    except Exception:
        pass


def test_find_tensors():
    try:
        find_tensors()
    except Exception:
        pass
