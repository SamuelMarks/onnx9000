from typing import Dict, List, Set, Tuple

from ..analysis import topological_sort
from ..expr import Call, Constant, Expr, Function, If, Let, Op, TupleExpr, TupleGetItem, Var
from ..ty import TensorType, TupleType


class MemoryPlanner:
    """
    Simulates a linear memory allocator for intermediate activations.
    """

    def __init__(self, alignment: int = 16):
        self.alignment = alignment
        self.offsets: dict[Expr, int] = {}
        self.sizes: dict[Expr, int] = {}
        self.total_size = 0

    def _get_dtype_size(self, dtype: str) -> int:
        if dtype in ("float32", "int32", "uint32"):
            return 4
        if dtype in ("float64", "int64", "uint64"):
            return 8
        if dtype in ("float16", "uint16", "int16"):
            return 2
        if dtype in ("uint8", "int8", "bool"):
            return 1
        return 4  # Default fallback

    def _align(self, size: int) -> int:
        return (size + self.alignment - 1) & ~(self.alignment - 1)

    def _compute_size(self, ty: TensorType) -> int:
        count = 1
        for dim in ty.shape:
            if isinstance(dim, str):
                # Dynamic shape can't be statically planned easily unless bounds are known.
                # Assuming upper bound or fallback to max size. For now, raise.
                raise ValueError("Dynamic shape cannot be statically memory planned")
            count *= dim
        return self._align(count * self._get_dtype_size(ty.dtype))

    def plan(self, expr: Expr) -> tuple[int, dict[Expr, int]]:
        """
        Plans memory and returns (total_size, offset_map)
        Currently implements a simple bump allocator without reuse.
        """
        nodes = topological_sort(expr)

        for node in nodes:
            if getattr(node, "checked_type", None) is None:
                continue

            ty = node.checked_type
            if isinstance(ty, TensorType):
                size = self._compute_size(ty)
                self.offsets[node] = self.total_size
                self.sizes[node] = size
                self.total_size += size

        return self.total_size, self.offsets


def plan_memory(expr: Expr) -> tuple[int, dict[Expr, int]]:
    """Performs explicit memory planning."""
    return MemoryPlanner().plan(expr)
