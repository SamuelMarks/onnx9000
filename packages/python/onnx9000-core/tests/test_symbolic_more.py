"""Tests the symbolic more module functionality."""

import ast

import pytest
from onnx9000.core.exceptions import ShapeInferenceError
from onnx9000.core.symbolic import (
    DynamicDim,
    broadcast_shapes,
    eval_expr,
    evaluate_symbolic_expression,
)


def test_symbolic_unresolved_name():
    """Tests the symbolic unresolved name functionality."""
    res = eval_expr(ast.parse("unknown", mode="eval").body, {})
    assert res == "unknown"


def test_symbolic_binop_mixed():
    """Tests the symbolic binop mixed functionality."""
    res = eval_expr(ast.parse("x + 1", mode="eval").body, {"x": "X"})
    assert "X" in res


def test_symbolic_unary_unresolved():
    """Tests the symbolic unary unresolved functionality."""
    res = eval_expr(ast.parse("-x", mode="eval").body, {"x": "X"})
    assert "-X" in res


def test_symbolic_type_error():
    """Tests the symbolic type error functionality."""
    with pytest.raises(Exception):
        eval_expr(ast.parse("True and False", mode="eval").body, {})


def test_evaluate_symbolic_expression_float_int():
    """Tests the evaluate symbolic expression float int functionality."""
    res = evaluate_symbolic_expression("4.0 / 2.0", {})
    assert res == 2
    assert isinstance(res, int)


def test_evaluate_symbolic_expression_exception():
    """Tests the evaluate symbolic expression exception functionality."""
    res = evaluate_symbolic_expression("x = 1", {})
    assert res == "x = 1"


def test_evaluate_symbolic_expression_in_context():
    """Tests the evaluate symbolic expression in context functionality."""
    res = evaluate_symbolic_expression("N", {"N": 5})
    assert res == 5


def test_broadcast_shapes_error():
    """Tests the broadcast shapes error functionality."""
    with pytest.raises(ShapeInferenceError):
        broadcast_shapes((2,), (3,))


def test_broadcast_shapes_unknowns():
    """Tests the broadcast shapes unknowns functionality."""
    assert broadcast_shapes((-1,), (3,)) == (3,)
    assert broadcast_shapes((3,), (-1,)) == (3,)


def test_broadcast_shapes_dynamic_strings():
    """Tests the broadcast shapes dynamic strings functionality."""
    res = broadcast_shapes((DynamicDim("N"),), (3,))
    assert res[0].value == "max(N, 3)"
    res2 = broadcast_shapes((3,), (DynamicDim("N"),))
    assert res2[0].value == "max(3, N)"


def test_broadcast_shapes_ones():
    """Tests the broadcast shapes ones functionality."""
    assert broadcast_shapes((1,), (3,)) == (3,)
    assert broadcast_shapes((3,), (1,)) == (3,)


def test_broadcast_shapes_equal():
    """Tests the broadcast shapes equal functionality."""
    assert broadcast_shapes((3,), (3,)) == (3,)


def test_symbolic_unary_resolved():
    """Tests the symbolic unary resolved functionality."""
    res = eval_expr(ast.parse("-5", mode="eval").body, {})
    assert res == -5


def test_symbolic_constant():
    """Tests the symbolic constant functionality."""
    res = eval_expr(ast.parse("10", mode="eval").body, {})
    assert res == 10


def test_evaluate_symbolic_expression_float_not_int():
    """Tests the evaluate symbolic expression float not int functionality."""
    res = evaluate_symbolic_expression("5.0 / 2.0", {})
    assert res == 2.5
    assert isinstance(res, float)


from onnx9000.core.symbolic import simplify_dim


def test_simplify_dim():
    """Tests the simplify dim functionality."""
    assert simplify_dim(DynamicDim("N")) == "N"
    assert simplify_dim(5) == 5
    assert simplify_dim("N") == "N"


def test_symbolic_type_error_else():
    """Tests the symbolic type error else functionality."""
    with pytest.raises(TypeError):
        eval_expr(ast.parse("lambda x: x", mode="eval").body, {})


def test_symbolic_constant_string():
    """Tests the symbolic constant string functionality."""
    res = eval_expr(ast.parse("'hello'", mode="eval").body, {})
    assert res == "hello"
