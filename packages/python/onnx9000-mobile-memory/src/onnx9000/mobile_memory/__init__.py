"""Mobile memory optimization and activation-checkpointing analysis for ONNX9000."""

from dataclasses import dataclass

from onnx9000.core.ir import Graph
from onnx9000.core.memory_planner import ArenaSimulator, MemoryBlock, simulate_memory_plan


@dataclass(frozen=True)
class MobileMemoryReport:
    """Summary of mobile peak memory usage and tensor offsets for an ONNX9000 `Graph`.

    Attributes:
        peak_bytes: Peak memory footprint in bytes.
        strategy: Allocation strategy used (`"first_fit"` or `"best_fit"`).
        tensor_offsets: Mapping of tensor names to byte offsets in the arena.
    """

    peak_bytes: int
    strategy: str
    tensor_offsets: dict[str, int]


class MobileMemory:
    """Analyzer and planner for optimizing ONNX9000 graph memory on mobile/edge devices."""

    def analyze(self, graph: Graph, strategy: str = "best_fit") -> MobileMemoryReport:
        """Simulate static arena allocation on `graph` and return a `MobileMemoryReport`.

        Args:
            graph: Target ONNX9000 `Graph`.
            strategy: Allocation strategy (`"first_fit"` or `"best_fit"`).

        Returns:
            Computed `MobileMemoryReport`.
        """
        arena = simulate_memory_plan(graph, strategy=strategy)
        return MobileMemoryReport(
            peak_bytes=arena.peak_memory,
            strategy=strategy,
            tensor_offsets=dict(arena.tensor_offsets),
        )

    def process(self, arg: str) -> str:
        """Process a mobile memory target identifier.

        Args:
            arg: Target identifier string.

        Returns:
            Formatted mobile memory status string.
        """
        return f"Mobile Memory processed {arg}"


__all__ = [
    "ArenaSimulator",
    "MemoryBlock",
    "MobileMemory",
    "MobileMemoryReport",
    "simulate_memory_plan",
]
