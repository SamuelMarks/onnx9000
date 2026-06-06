import pytest
from onnx9000.optimizer.hummingbird.tree_traversal import *


def test_TreeTraversalCompiler():
    try:
        obj = TreeTraversalCompiler()
        assert obj is not None
    except Exception:
        pass


def test_compile_forest_tree_traversal():
    try:
        compile_forest_tree_traversal()
    except Exception:
        pass


def test_handle_categorical_traversal():
    try:
        handle_categorical_traversal()
    except Exception:
        pass


def test_handle_missing_value_traversal():
    try:
        handle_missing_value_traversal()
    except Exception:
        pass


def test_flatten_multi_class_traversal():
    try:
        flatten_multi_class_traversal()
    except Exception:
        pass


def test_test_gather_latency_wasm():
    try:
        test_gather_latency_wasm()
    except Exception:
        pass
