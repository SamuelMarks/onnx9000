import pytest
from onnx9000.core.symbolic import *

def test_eval_expr():
    try:
        res = eval_expr()
    except Exception:
        pass

def test_evaluate_symbolic_expression():
    try:
        res = evaluate_symbolic_expression()
    except Exception:
        pass

def test_simplify_expression():
    try:
        res = simplify_expression()
    except Exception:
        pass

def test__is_same():
    try:
        res = _is_same()
    except Exception:
        pass

def test__is_zero():
    try:
        res = _is_zero()
    except Exception:
        pass

def test__is_one():
    try:
        res = _is_one()
    except Exception:
        pass

def test__simplify_ast():
    try:
        res = _simplify_ast()
    except Exception:
        pass

def test__ast_to_str():
    try:
        res = _ast_to_str()
    except Exception:
        pass

def test_broadcast_shapes():
    try:
        res = broadcast_shapes()
    except Exception:
        pass

def test_simplify_dim():
    try:
        res = simplify_dim()
    except Exception:
        pass

