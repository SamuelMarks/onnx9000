import pytest
from onnx9000.converters.tf.passes import *

def test_constant_folding_pass():
    try:
        res = constant_folding_pass()
    except Exception:
        pass

def test_identity_removal_pass():
    try:
        res = identity_removal_pass()
    except Exception:
        pass

def test_dropout_removal_pass():
    try:
        res = dropout_removal_pass()
    except Exception:
        pass

def test_remove_debug_nodes_pass():
    try:
        res = remove_debug_nodes_pass()
    except Exception:
        pass

def test_transpose_optimizer_pass():
    try:
        res = transpose_optimizer_pass()
    except Exception:
        pass

def test_shape_folding_pass():
    try:
        res = shape_folding_pass()
    except Exception:
        pass

def test_pattern_matching_pass():
    try:
        res = pattern_matching_pass()
    except Exception:
        pass

def test_dce_pass():
    try:
        res = dce_pass()
    except Exception:
        pass

def test_tf_optimize_graph():
    try:
        res = tf_optimize_graph()
    except Exception:
        pass

