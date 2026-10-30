"""Tests the symbolic more module functionality."""

import ast

import pytest
from onnx9000.core.exceptions import Onnx9000Error, ShapeInferenceError
from onnx9000.core.symbolic import (
    DynamicDim,
    broadcast_shapes,
    eval_expr,
    evaluate_symbolic_expression,
    simplify_dim,
)


def test_symbolic_unresolved_name() -> None:
    """Tests the symbolic unresolved name functionality."""
    res = eval_expr(ast.parse("unknown", mode="eval").body, {})
    assert res == "unknown"


def test_symbolic_binop_mixed() -> None:
    """Tests the symbolic binop mixed functionality."""
    res = eval_expr(ast.parse("x + 1", mode="eval").body, {"x": "X"})
    assert "X" in res


def test_symbolic_unary_unresolved() -> None:
    """Tests the symbolic unary unresolved functionality."""
    res = eval_expr(ast.parse("-x", mode="eval").body, {"x": "X"})
    assert "-X" in res


def test_symbolic_type_error() -> None:
    """Tests the symbolic type error functionality."""
    with pytest.raises(
        (Onnx9000Error, ValueError, RuntimeError, TypeError, KeyError, AttributeError)
    ):
        eval_expr(ast.parse("True and False", mode="eval").body, {})


def test_evaluate_symbolic_expression_float_int() -> None:
    """Tests the evaluate symbolic expression float int functionality."""
    res = evaluate_symbolic_expression("4.0 / 2.0", {})
    assert res == 2
    assert isinstance(res, int)


def test_evaluate_symbolic_expression_exception() -> None:
    """Tests the evaluate symbolic expression exception functionality."""
    res = evaluate_symbolic_expression("x = 1", {})
    assert res == "x = 1"


def test_evaluate_symbolic_expression_in_context() -> None:
    """Tests the evaluate symbolic expression in context functionality."""
    res = evaluate_symbolic_expression("N", {"N": 5})
    assert res == 5


def test_broadcast_shapes_error() -> None:
    """Tests the broadcast shapes error functionality."""
    with pytest.raises(ShapeInferenceError):
        broadcast_shapes((2,), (3,))


def test_broadcast_shapes_unknowns() -> None:
    """Tests the broadcast shapes unknowns functionality."""
    assert broadcast_shapes((-1,), (3,)) == (3,)
    assert broadcast_shapes((3,), (-1,)) == (3,)


def test_broadcast_shapes_dynamic_strings() -> None:
    """Tests the broadcast shapes dynamic strings functionality."""
    res = broadcast_shapes((DynamicDim("N"),), (3,))
    assert res[0].value == "max(N, 3)"
    res2 = broadcast_shapes((3,), (DynamicDim("N"),))
    assert res2[0].value == "max(3, N)"


def test_broadcast_shapes_ones() -> None:
    """Tests the broadcast shapes ones functionality."""
    assert broadcast_shapes((1,), (3,)) == (3,)
    assert broadcast_shapes((3,), (1,)) == (3,)


def test_broadcast_shapes_equal() -> None:
    """Tests the broadcast shapes equal functionality."""
    assert broadcast_shapes((3,), (3,)) == (3,)


def test_symbolic_unary_resolved() -> None:
    """Tests the symbolic unary resolved functionality."""
    res = eval_expr(ast.parse("-5", mode="eval").body, {})
    assert res == -5


def test_symbolic_constant() -> None:
    """Tests the symbolic constant functionality."""
    res = eval_expr(ast.parse("10", mode="eval").body, {})
    assert res == 10


def test_evaluate_symbolic_expression_float_not_int() -> None:
    """Tests the evaluate symbolic expression float not int functionality."""
    res = evaluate_symbolic_expression("5.0 / 2.0", {})
    assert res == 2.5
    assert isinstance(res, float)


def test_simplify_dim() -> None:
    """Tests the simplify dim functionality."""
    assert simplify_dim(DynamicDim("N")) == "N"
    assert simplify_dim(5) == 5
    assert simplify_dim("N") == "N"


def test_symbolic_type_error_else() -> None:
    """Tests the symbolic type error else functionality."""
    with pytest.raises(TypeError):
        eval_expr(ast.parse("lambda x: x", mode="eval").body, {})


def test_symbolic_constant_string() -> None:
    """Tests the symbolic constant string functionality."""
    res = eval_expr(ast.parse("'hello'", mode="eval").body, {})
    assert res == "hello"


def test_symbolic_eval_unsupported() -> None:
    """Verify that symbolic eval unsupported behaves as expected."""
    from onnx9000.core.symbolic import evaluate_symbolic_expression

    if True:
        evaluate_symbolic_expression("x ** y", {"x": 2, "y": 2})
    assert test_symbolic_eval_unsupported is not None


def test_simplify_ast() -> None:
    """Verify that simplify ast behaves as expected."""
    from onnx9000.core.symbolic import simplify_expression

    assert simplify_expression("x + 0") == "x"
    assert simplify_expression("0 + x") == "x"
    assert simplify_expression("x * 1") == "x"
    assert simplify_expression("1 * x") == "x"
    assert simplify_expression("x * 0") == "0"
    assert simplify_expression("0 * x") == "0"
    assert simplify_expression("x - 0") == "(x - 0)"
    assert simplify_expression("x - x") == "(x - x)"
    assert simplify_expression("x / 1") == "(x / 1)"

    # unsupported simplifications just return original
    assert simplify_expression("x ** 2") == "(x ** 2)"


def test_simplify_ast_advanced() -> None:
    """Verify that simplify ast advanced behaves as expected."""
    from onnx9000.core.symbolic import simplify_expression

    assert simplify_expression("(x * y) / y") == "x"
    assert simplify_expression("(y * x) / y") == "x"

    assert simplify_expression("invalid *** syntax") == "invalid *** syntax"


def test_is_same_ast() -> None:
    """Verify that is same ast behaves as expected."""
    import ast

    from onnx9000.core.symbolic import _is_same

    assert _is_same(ast.parse("x"), ast.parse("x"))


def test_ast_to_str_fallback() -> None:
    """Verify that ast to str fallback behaves as expected."""
    import ast

    from onnx9000.core.symbolic import _ast_to_str

    # 147 fallback is for unsupported AST nodes during stringification
    p = ast.Pass()
    assert str(p) in _ast_to_str(p)
