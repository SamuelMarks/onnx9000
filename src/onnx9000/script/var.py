"""
Defines symbolic variables that represent edges and tensors within an ONNX graph.
These variables provide operator overloading to allow intuitive mathematical and logical expression construction.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import builtins
import numpy as np
from onnx9000.core.ir import Node

# Global counter for unique naming
_global_name_counter = 0


def _generate_unique_name(prefix: str = "tmp") -> str:
    """Generates a globally unique identifier with the given prefix."""
    global _global_name_counter
    _global_name_counter += 1
    return f"{prefix}_{_global_name_counter}"


class Var:
    """Represents a symbolic edge (tensor) connecting nodes in the ONNX graph."""

    def __init__(self, name: Optional[str] = None):
        """Initializes the symbolic variable with an optional specific name, otherwise auto-generates one."""
        self.name: str = name if name is not None else _generate_unique_name("var")

    def rename(self, new_name: str) -> "Var":
        """Updates the name of this variable, modifying it in place."""
        self.name = new_name
        return self

    def __repr__(self) -> str:
        """Returns the string representation of the variable instance."""
        return f"Var({self.name})"

    def __add__(self, other: Any) -> "Var":
        """Builds an addition operation between this variable and another operand."""
        return _op_instance.Add(self, other)

    def __radd__(self, other: Any) -> "Var":
        """Builds an addition operation between another operand and this variable."""
        return _op_instance.Add(other, self)

    def __sub__(self, other: Any) -> "Var":
        """Builds a subtraction operation where the other operand is subtracted from this variable."""
        return _op_instance.Sub(self, other)

    def __rsub__(self, other: Any) -> "Var":
        """Builds a subtraction operation where this variable is subtracted from the other operand."""
        return _op_instance.Sub(other, self)

    def __mul__(self, other: Any) -> "Var":
        """Builds a multiplication operation between this variable and another operand."""
        return _op_instance.Mul(self, other)

    def __rmul__(self, other: Any) -> "Var":
        """Builds a multiplication operation between another operand and this variable."""
        return _op_instance.Mul(other, self)

    def __truediv__(self, other: Any) -> "Var":
        """Builds a division operation where this variable is divided by the other operand."""
        return _op_instance.Div(self, other)

    def __rtruediv__(self, other: Any) -> "Var":
        """Builds a division operation where the other operand is divided by this variable."""
        return _op_instance.Div(other, self)

    def __pow__(self, other: Any) -> "Var":
        """Builds a power operation raising this variable to the exponent given by the other operand."""
        return _op_instance.Pow(self, other)

    def __matmul__(self, other: Any) -> "Var":
        """Builds a matrix multiplication operation between this variable and another operand."""
        return _op_instance.MatMul(self, other)

    def __gt__(self, other: Any) -> "Var":
        """Builds a strictly greater-than comparison operation."""
        return _op_instance.Greater(self, other)

    def __lt__(self, other: Any) -> "Var":
        """Builds a strictly less-than comparison operation."""
        return _op_instance.Less(self, other)

    def __eq__(self, other: Any) -> "Var":  # type: ignore
        """Builds an equality comparison operation between this variable and another operand."""
        return _op_instance.Equal(self, other)

    def __ne__(self, other: Any) -> "Var":  # type: ignore
        """Builds a non-equality comparison operation between this variable and another operand."""
        # Since NotEqual opset is newer, might need to composite or use NotEqual
        return _op_instance.NotEqual(self, other)

    def __and__(self, other: Any) -> "Var":
        """Builds a bitwise or logical AND operation between this variable and another operand."""
        return _op_instance.BitwiseAnd(self, other)

    def __or__(self, other: Any) -> "Var":
        """Builds a bitwise or logical OR operation between this variable and another operand."""
        return _op_instance.BitwiseOr(self, other)

    def __xor__(self, other: Any) -> "Var":
        """Builds a bitwise or logical XOR operation between this variable and another operand."""
        return _op_instance.BitwiseXor(self, other)

    def __invert__(self) -> "Var":
        """Builds a bitwise or logical NOT operation on this variable."""
        return _op_instance.BitwiseNot(self)

    def __getitem__(self, idx: Any) -> "Var":
        """Builds either a slicing or gathering operation based on the indexing structure."""
        if isinstance(idx, builtins.slice):
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else 2147483647
            step = idx.step if idx.step is not None else 1
            # In a real graph we would need to generate constant nodes for start, stop, axes, steps
            # but we can rely on _op_instance.Slice implementation to handle python ints
            return _op_instance.Slice(
                self, start, stop, 0, step
            )  # Assuming axis 0 for simplistic slice
        else:
            return _op_instance.Gather(self, idx, axis=0)


# Import at end to avoid circular dependency
from onnx9000.script.op import op as _op_instance
