"""Unit tests for symbolic dimension evaluation in `onnx9000.core.symbolic`."""

import ast

import pytest
from onnx9000.core.exceptions import ShapeInferenceError
from onnx9000.core.ir import DynamicDim
from onnx9000.core.symbolic import (
    _ast_to_str,
    _is_one,
    _is_same,
    _is_zero,
    _simplify_ast,
    broadcast_shapes,
    eval_expr,
    evaluate_symbolic_expression,
    simplify_dim,
    simplify_expression,
)


def test_symbolic_evaluation_and_broadcasting() -> None:
    """Verify eval_expr, evaluate_symbolic_expression, simplify_expression, and broadcast_shapes."""
    tree = ast.parse("x + 2 * y", mode="eval").body
    assert eval_expr(tree, {"x": 3, "y": 4}) == 11

    assert evaluate_symbolic_expression("N * 2", {"N": 8}) == 16
    assert evaluate_symbolic_expression("N + M", {"N": 8}) == "8 + M"

    assert simplify_expression("(x * y) / y") == "x"
    assert simplify_expression("x + 0") == "x"
    assert simplify_expression("x * 1") == "x"
    assert simplify_expression("x * 0") == "0"

    c0 = ast.Constant(0)
    c1 = ast.Constant(1)
    assert _is_zero(c0) is True
    assert _is_one(c1) is True
    assert _is_same(c0, ast.Constant(0)) is True

    simp = _simplify_ast(ast.parse("a + 0", mode="eval").body)
    assert _ast_to_str(simp) == "a"

    out_shape = broadcast_shapes((1, 4), (3, 1))
    assert out_shape == (3, 4)

    with pytest.raises(ShapeInferenceError):
        broadcast_shapes((2, 3), (4, 5))

    assert simplify_dim(DynamicDim("B")) == "B"
    assert simplify_dim(16) == 16
