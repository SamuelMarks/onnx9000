"""Symbolic mathematical evaluation module for ONNX shapes."""

import ast
import operator as op
from typing import Any, Union

from onnx9000.core.exceptions import ShapeInferenceError
from onnx9000.core.ir import DynamicDim

# Supported operators
operators = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}


def eval_expr(node: Any, context: dict[str, int]) -> Union[int, float, str]:
    """Execute the eval expr operation."""
    if isinstance(node, ast.Name):
        if node.id in context:
            return context[node.id]
        return node.id
    elif isinstance(node, ast.BinOp):
        left = eval_expr(node.left, context)
        right = eval_expr(node.right, context)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return operators[type(node.op)](left, right)
        op_str_map = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.FloorDiv: "//",
            ast.Mod: "%",
            ast.Pow: "**",
        }
        op_str = op_str_map.get(type(node.op), type(node.op).__name__)
        return f"{left} {op_str} {right}"
    elif isinstance(node, ast.UnaryOp):
        operand = eval_expr(node.operand, context)
        if isinstance(operand, (int, float)):
            return operators[type(node.op)](operand)
        return f"(-{operand})"
    elif isinstance(node, ast.Constant):
        return node.value
    else:
        raise TypeError(node)


def evaluate_symbolic_expression(expr: str, context: dict[str, int]) -> Union[int, str]:
    """Evaluate a symbolic string expression given a context mapping variables to integers.

    Fallback to string if unresolved.
    """
    if expr in context:
        return context[expr]
    try:
        node = ast.parse(expr, mode="eval").body
        res = eval_expr(node, context)
        if isinstance(res, float) and res.is_integer():
            return int(res)
        return res
    except Exception:
        return expr


def simplify_expression(expr: str) -> str:
    """Simplify a symbolic string expression (e.g. '(x * y) / y' -> 'x')."""
    try:
        node = ast.parse(expr, mode="eval").body
        simplified = _simplify_ast(node)
        return _ast_to_str(simplified)
    except Exception:
        return expr


def _is_same(n1: Any, n2: Any) -> bool:
    """Check if two AST nodes are structurally identical."""
    return ast.dump(n1) == ast.dump(n2)


def _is_zero(node: Any) -> bool:
    """Check if an AST node is a constant zero."""
    return isinstance(node, ast.Constant) and node.value == 0


def _is_one(node: Any) -> bool:
    """Check if an AST node is a constant one."""
    return isinstance(node, ast.Constant) and node.value == 1


def _simplify_ast(node: Any) -> Any:
    """Recursively simplify an AST node."""
    if isinstance(node, ast.BinOp):
        left = _simplify_ast(node.left)
        right = _simplify_ast(node.right)

        # (x * y) / y -> x
        if isinstance(node.op, ast.Div):
            if isinstance(left, ast.BinOp) and isinstance(left.op, ast.Mult):
                if _is_same(left.right, right):
                    return left.left
                if _is_same(left.left, right):
                    return left.right

        # x + 0 -> x, x * 1 -> x, etc.
        if isinstance(node.op, ast.Add):
            if _is_zero(right):
                return left
            if _is_zero(left):
                return right
        if isinstance(node.op, ast.Mult):
            if _is_one(right):
                return left
            if _is_one(left):
                return right
            if _is_zero(right) or _is_zero(left):
                return ast.Constant(0)

        return ast.BinOp(left, node.op, right)
    return node


def _ast_to_str(node: Any) -> str:
    """Convert a simplified AST node back to a string."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant):
        return str(node.value)
    if isinstance(node, ast.BinOp):
        op_map = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.FloorDiv: "//",
            ast.Mod: "%",
            ast.Pow: "**",
        }
        return f"({_ast_to_str(node.left)} {op_map[type(node.op)]} {_ast_to_str(node.right)})"
    return str(node)


def broadcast_shapes(
    shape_a: tuple[Union[int, DynamicDim], ...],
    shape_b: tuple[Union[int, DynamicDim], ...],
) -> tuple[Union[int, DynamicDim], ...]:
    """Apply standard NumPy broadcasting rules to two shapes, including dynamic dimensions."""
    max_len = max(len(shape_a), len(shape_b))
    padded_a = (1,) * (max_len - len(shape_a)) + shape_a
    padded_b = (1,) * (max_len - len(shape_b)) + shape_b
    result = []
    for a, b in zip(padded_a, padded_b):
        a_val = a.value if isinstance(a, DynamicDim) else a
        b_val = b.value if isinstance(b, DynamicDim) else b

        # If one of them is 1, broadcast to the other
        if a_val == 1:
            result.append(b)
        elif b_val == 1:
            result.append(a)
        # If they are exactly equal, take either
        elif a_val == b_val:
            result.append(a)
        # Handle string symbols
        elif isinstance(a_val, str) or isinstance(b_val, str):
            result.append(DynamicDim(f"max({a_val}, {b_val})"))
        # Handle unknown dimensions (-1)
        elif a_val == -1:
            result.append(b)
        elif b_val == -1:
            result.append(a)
        else:
            raise ShapeInferenceError(
                f"Operands could not be broadcast together with shapes {shape_a} {shape_b}"
            )
    return tuple(result)


def simplify_dim(dim: Union[int, DynamicDim, str]) -> Union[int, str]:
    """Execute the simplify dim operation."""
    if isinstance(dim, DynamicDim):
        return dim.value
    return dim
