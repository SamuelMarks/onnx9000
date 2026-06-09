"""Static Arena Allocator."""

from ml_switcheroo_ir import LogicalGraph, topological_sort


class ArenaAllocator:
    def __init__(self, page_size: int = 65536):
        self.page_size = page_size
        self.offsets: dict[str, int] = {}
        self.total_bytes = 0

    def allocate(self, graph: LogicalGraph) -> tuple[dict[str, int], int]:
        """Calculates exact lifetime and byte-offsets for every tensor.

        Args:
            graph: The logical graph.

        Returns:
            Tuple containing:
            - Dict mapping node IDs to byte offsets.
            - Total bytes allocated.
        """
        self.offsets = {}
        self.total_bytes = 0

        # Simple bump allocator for now
        # A real implementation would compute liveness intervals and reuse memory.
        import numpy as np

        dtype_sizes = {
            "float32": 4,
            "float16": 2,
            "bfloat16": 2,
            "int64": 8,
            "int32": 4,
            "bool": 1,
        }

        for node in topological_sort(graph):
            shape = node.shape_metadata
            if shape is None:
                # Fallback to scalar
                num_elements = 1
            else:
                num_elements = 1
                for dim in shape:
                    if isinstance(dim, int):
                        num_elements *= dim
                    else:
                        num_elements *= 1  # Unknown dynamic dim, mocked as 1

            # Just default to float32 size
            dtype = node.attributes.get("dtype", "float32")
            size_bytes = num_elements * dtype_sizes.get(dtype, 4)

            # Align to 8 bytes
            size_bytes = (size_bytes + 7) & ~7

            self.offsets[node.id] = self.total_bytes
            self.total_bytes += size_bytes

        return self.offsets, self.total_bytes

    def get_wasm_pages(self) -> int:
        """Returns the number of WASM memory pages required."""
        return (self.total_bytes + self.page_size - 1) // self.page_size
