"""MACs, FLOPs, parameter, and memory bandwidth profiler tool (`ONNXTool`) for ONNX9000."""

from onnx9000.core.ir import Graph
from onnx9000.core.profiler import ProfilerResult, dtype_size, profile, resolve_volume


class ONNXTool:
    """High-level diagnostic profiler for analyzing ONNX9000 graphs."""

    def profile_graph(
        self, graph: Graph, dynamic_overrides: dict[str, int] | None = None
    ) -> ProfilerResult:
        """Compute MACs, FLOPs, parameters, and activation bytes for `graph`.

        Args:
            graph: Target ONNX9000 `Graph`.
            dynamic_overrides: Optional mapping of symbolic dimension names to concrete ints.

        Returns:
            Populated `ProfilerResult`.
        """
        return profile(graph, dynamic_overrides=dynamic_overrides)

    def process(self, arg: str) -> str:
        """Process a model target identifier.

        Args:
            arg: Target identifier string.

        Returns:
            Formatted ONNX Tool status string.
        """
        return f"ONNX Tool processed {arg}"


__all__ = ["ONNXTool", "ProfilerResult", "dtype_size", "profile", "resolve_volume"]
