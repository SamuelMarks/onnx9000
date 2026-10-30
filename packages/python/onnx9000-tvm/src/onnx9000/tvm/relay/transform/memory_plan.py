"""Relay static memory planner pass."""

from onnx9000.core.exceptions import CompilationError

from ..analysis import topological_sort
from ..expr import Expr
from ..ty import TensorType


class MemoryPlanner:
    """Simulates a linear memory allocator for intermediate activations."""

    def __init__(self, alignment: int = 16) -> None:
        """Initialize the memory planner with a byte alignment."""
        self.alignment = alignment
        self.offsets: dict[Expr, int] = {}
        self.sizes: dict[Expr, int] = {}
        self.total_size = 0

    def _get_dtype_size(self, dtype: str) -> int:
        """Return the byte width of a tensor dtype."""
        if dtype in ("float32", "int32", "uint32"):
            return 4
        if dtype in ("float64", "int64", "uint64"):
            return 8
        if dtype in ("float16", "uint16", "int16"):
            return 2
        if dtype in ("uint8", "int8", "bool"):
            return 1
        return 4

    def _align(self, size: int) -> int:
        """Align a byte size to the configured alignment boundary."""
        return (size + self.alignment - 1) & ~(self.alignment - 1)

    def _compute_size(self, ty: TensorType) -> int:
        """Compute the aligned byte size of a TensorType."""
        count = 1
        for dim in ty.shape:
            if isinstance(dim, str):
                raise CompilationError("Dynamic shape cannot be statically memory planned")
            count *= dim
        return self._align(count * self._get_dtype_size(ty.dtype))

    def plan(self, expr: Expr) -> tuple[int, dict[Expr, int]]:
        """Plan memory and return (total_size, offset_map)."""
        nodes = topological_sort(expr)

        for node in nodes:
            if node.checked_type is None:
                continue

            ty = node.checked_type
            if isinstance(ty, TensorType):
                size = self._compute_size(ty)
                self.offsets[node] = self.total_size
                self.sizes[node] = size
                self.total_size += size

        return self.total_size, self.offsets


def plan_memory(expr: Expr) -> tuple[int, dict[Expr, int]]:
    """Perform explicit memory planning."""
    return MemoryPlanner().plan(expr)
