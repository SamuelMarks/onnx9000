"""Symbolic mathematical evaluation module for ONNX shapes."""

from typing import Union
from onnx9000.core.exceptions import ShapeInferenceError
from onnx9000.core.ir import DynamicDim


def evaluate_symbolic_expression(expr: str, context: dict[str, int]) -> Union[int, str]:
    """
    Evaluates a symbolic string expression given a context mapping variables to integers.
    Fallback to string if unresolved.
    """
    if expr in context:
        return context[expr]
    return expr


def broadcast_shapes(
    shape_a: tuple[Union[int, DynamicDim], ...], shape_b: tuple[Union[int, DynamicDim], ...]
) -> tuple[Union[int, DynamicDim], ...]:
    """
    Applies standard NumPy broadcasting rules to two shapes, including dynamic dimensions.
    """
    max_len = max(len(shape_a), len(shape_b))
    padded_a = (1,) * (max_len - len(shape_a)) + shape_a
    padded_b = (1,) * (max_len - len(shape_b)) + shape_b
    result = []
    for a, b in zip(padded_a, padded_b):
        a_val = a.value if isinstance(a, DynamicDim) else a
        b_val = b.value if isinstance(b, DynamicDim) else b
        if a_val == b_val:
            result.append(a)
        elif a_val == 1:
            result.append(b)
        elif b_val == 1:
            result.append(a)
        elif isinstance(a_val, str) or isinstance(b_val, str):
            result.append(DynamicDim(f"max({a_val}, {b_val})"))
        elif a_val == -1:
            result.append(b)
        elif b_val == -1:
            result.append(a)
        else:
            raise ShapeInferenceError(
                f"Operands could not be broadcast together with shapes {shape_a} {shape_b}"
            )
    return tuple(result)
