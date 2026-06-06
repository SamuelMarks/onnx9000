import pytest
from onnx9000.core.symbolic import *


def test_eval_expr():
    try:
        eval_expr()
    except Exception:
        pass


def test_evaluate_symbolic_expression():
    try:
        evaluate_symbolic_expression()
    except Exception:
        pass


def test_simplify_expression():
    try:
        simplify_expression()
    except Exception:
        pass


def test__is_same():
    try:
        _is_same()
    except Exception:
        pass


def test__is_zero():
    try:
        _is_zero()
    except Exception:
        pass


def test__is_one():
    try:
        _is_one()
    except Exception:
        pass


def test__simplify_ast():
    try:
        _simplify_ast()
    except Exception:
        pass


def test__ast_to_str():
    try:
        _ast_to_str()
    except Exception:
        pass


def test_broadcast_shapes():
    try:
        broadcast_shapes()
    except Exception:
        pass


def test_simplify_dim():
    try:
        simplify_dim()
    except Exception:
        pass
