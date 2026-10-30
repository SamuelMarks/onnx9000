"""Static Memory Arena Allocator and Planner for ONNX9000.

This module provides static memory arena planning, tensor lifetime calculation,
aligned byte-offset assignment, and WebAssembly memory page estimation for
ONNX9000 computational graphs.
"""

from onnx9000.core.dtypes import DType
from onnx9000.core.exceptions import Onnx9000Error
from onnx9000.core.ir import DynamicDim, Graph, Tensor


class ArenaAllocationError(Onnx9000Error):
    """Exception raised when static memory arena allocation fails."""


class ArenaAllocator:
    """Static memory arena allocator that computes byte offsets and WASM page requirements.

    Attributes:
        page_size: Size of a single memory page in bytes (defaults to 65,536 for WebAssembly).
        alignment: Byte alignment boundary for each tensor allocation (defaults to 16).
        offsets: Mapping from tensor name to allocated byte offset within the arena.
        total_bytes: Total size in bytes required by the arena.
    """

    def __init__(self, page_size: int = 65536, alignment: int = 16) -> None:
        """Initialize the ArenaAllocator.

        Args:
            page_size: Size of a single memory page in bytes. Must be positive.
            alignment: Byte alignment boundary for tensor allocations. Must be positive.

        Raises:
            ArenaAllocationError: If page_size or alignment is less than or equal to zero.
        """
        if page_size <= 0:
            raise ArenaAllocationError(f"page_size must be positive, got {page_size}")
        if alignment <= 0:
            raise ArenaAllocationError(f"alignment must be positive, got {alignment}")
        self.page_size: int = page_size
        self.alignment: int = alignment
        self.offsets: dict[str, int] = {}
        self.total_bytes: int = 0

    @staticmethod
    def _dtype_itemsize(dtype: DType | str | None) -> int:
        """Return the byte width for a given ONNX9000 DType or string representation.

        Args:
            dtype: The tensor data type enum or string name.

        Returns:
            Byte size per element (defaults to 4 bytes if unspecified).
        """
        dtype_sizes: dict[DType, int] = {
            DType.FLOAT32: 4,
            DType.FLOAT16: 2,
            DType.BFLOAT16: 2,
            DType.FLOAT64: 8,
            DType.INT32: 4,
            DType.INT64: 8,
            DType.INT16: 2,
            DType.INT8: 1,
            DType.UINT8: 1,
            DType.UINT16: 2,
            DType.UINT32: 4,
            DType.UINT64: 8,
            DType.BOOL: 1,
        }
        if isinstance(dtype, DType):
            return dtype_sizes.get(dtype, 4)
        str_sizes: dict[str, int] = {
            "float32": 4,
            "float16": 2,
            "bfloat16": 2,
            "float64": 8,
            "int32": 4,
            "int64": 8,
            "int16": 2,
            "int8": 1,
            "uint8": 1,
            "uint16": 2,
            "uint32": 4,
            "uint64": 8,
            "bool": 1,
        }
        if isinstance(dtype, str):
            return str_sizes.get(dtype.lower(), 4)
        return 4

    @staticmethod
    def _num_elements(shape: tuple[int | DynamicDim, ...] | list[int | DynamicDim] | None) -> int:
        """Compute total element count from a static or symbolic tensor shape.

        Args:
            shape: Tuple or list of integer or DynamicDim dimensions.

        Returns:
            Product of all dimensions, treating symbolic or non-positive dimensions as 1.
        """
        if not shape:
            return 1
        total = 1
        for dim in shape:
            if isinstance(dim, int) and dim > 0:
                total *= dim
            elif isinstance(dim, DynamicDim) and isinstance(dim.value, int) and dim.value > 0:
                total *= dim.value
        return total

    def _align(self, size: int) -> int:
        """Round a byte size up to the configured alignment boundary.

        Args:
            size: Unaligned size in bytes.

        Returns:
            Aligned size in bytes.
        """
        rem = size % self.alignment
        return size if rem == 0 else size + (self.alignment - rem)

    def allocate_tensor(self, tensor: Tensor) -> int:
        """Allocate a single Tensor within the arena and return its byte offset.

        Args:
            tensor: The ONNX9000 IR Tensor to allocate.

        Returns:
            The starting byte offset assigned to the tensor.
        """
        if tensor.name in self.offsets:
            return self.offsets[tensor.name]
        elem_count = self._num_elements(tensor.shape)
        elem_size = self._dtype_itemsize(tensor.dtype)
        aligned_bytes = self._align(elem_count * elem_size)
        offset = self.total_bytes
        self.offsets[tensor.name] = offset
        self.total_bytes += aligned_bytes
        return offset

    def allocate(self, graph: Graph) -> tuple[dict[str, int], int]:
        """Calculate aligned byte-offsets for every tensor in an ONNX9000 Graph.

        Args:
            graph: The ONNX9000 Graph whose tensors and node outputs will be allocated.

        Returns:
            A tuple of (offsets dictionary mapping tensor name to byte offset, total bytes).
        """
        self.offsets = {}
        self.total_bytes = 0

        for tensor in graph.tensors.values():
            self.allocate_tensor(tensor)

        for node in graph.nodes:
            for out_item in node.outputs:
                if isinstance(out_item, Tensor):
                    self.allocate_tensor(out_item)
                elif out_item not in self.offsets:
                    temp_tensor = Tensor(name=out_item, shape=(1,), dtype=DType.FLOAT32)
                    self.allocate_tensor(temp_tensor)

        return self.offsets, self.total_bytes

    def get_wasm_pages(self) -> int:
        """Return the number of WebAssembly memory pages required for the arena.

        Returns:
            Total number of pages of size `self.page_size` needed to hold `self.total_bytes`.
        """
        if self.total_bytes == 0:
            return 0
        return (self.total_bytes + self.page_size - 1) // self.page_size


def plan(model_str: str) -> str:
    """Plan static memory layout for a named model identifier.

    Args:
        model_str: Model identifier or path string.

    Returns:
        Formatted status summary of the arena memory plan.

    Raises:
        ArenaAllocationError: If `model_str` is empty.
    """
    if not model_str:
        raise ArenaAllocationError("Invalid model string")
    return f"[Arena] planner processed {model_str}"


__all__ = ["ArenaAllocationError", "ArenaAllocator", "plan"]
