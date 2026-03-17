import pytest
import ast
from onnx9000.core.symbolic import (
    eval_expr,
    evaluate_symbolic_expression,
    broadcast_shapes,
    DynamicDim,
)
from onnx9000.core.exceptions import ShapeInferenceError


def test_symbolic_unresolved_name():
    res = eval_expr(ast.parse("unknown", mode="eval").body, {})
    assert res == "unknown"


def test_symbolic_binop_mixed():
    res = eval_expr(ast.parse("x + 1", mode="eval").body, {"x": "X"})
    assert "X" in res


def test_symbolic_unary_unresolved():
    res = eval_expr(ast.parse("-x", mode="eval").body, {"x": "X"})
    assert "-X" in res


def test_symbolic_type_error():
    with pytest.raises(Exception):
        eval_expr(ast.parse("True and False", mode="eval").body, {})


def test_evaluate_symbolic_expression_float_int():
    res = evaluate_symbolic_expression("4.0 / 2.0", {})
    assert res == 2
    assert isinstance(res, int)


def test_evaluate_symbolic_expression_exception():
    res = evaluate_symbolic_expression("x = 1", {})
    assert res == "x = 1"


def test_evaluate_symbolic_expression_in_context():
    res = evaluate_symbolic_expression("N", {"N": 5})
    assert res == 5


def test_broadcast_shapes_error():
    with pytest.raises(ShapeInferenceError):
        broadcast_shapes((2,), (3,))


def test_broadcast_shapes_unknowns():
    assert broadcast_shapes((-1,), (3,)) == (3,)
    assert broadcast_shapes((3,), (-1,)) == (3,)


def test_broadcast_shapes_dynamic_strings():
    res = broadcast_shapes((DynamicDim("N"),), (3,))
    assert res[0].value == "max(N, 3)"
    res2 = broadcast_shapes((3,), (DynamicDim("N"),))
    assert res2[0].value == "max(3, N)"


def test_broadcast_shapes_ones():
    assert broadcast_shapes((1,), (3,)) == (3,)
    assert broadcast_shapes((3,), (1,)) == (3,)


def test_broadcast_shapes_equal():
    assert broadcast_shapes((3,), (3,)) == (3,)


def test_symbolic_unary_resolved():
    res = eval_expr(ast.parse("-5", mode="eval").body, {})
    assert res == -5


def test_symbolic_constant():
    res = eval_expr(ast.parse("10", mode="eval").body, {})
    assert res == 10


def test_evaluate_symbolic_expression_float_not_int():
    res = evaluate_symbolic_expression("5.0 / 2.0", {})
    assert res == 2.5
    assert isinstance(res, float)


from onnx9000.core.symbolic import simplify_dim


def test_simplify_dim():
    assert simplify_dim(DynamicDim("N")) == "N"
    assert simplify_dim(5) == 5
    assert simplify_dim("N") == "N"


def test_symbolic_type_error_else():
    with pytest.raises(TypeError):
        eval_expr(ast.parse("lambda x: x", mode="eval").body, {})


def test_symbolic_constant_string():
    res = eval_expr(ast.parse("'hello'", mode="eval").body, {})
    assert res == "hello"
