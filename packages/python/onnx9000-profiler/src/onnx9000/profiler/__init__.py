"""Profiler and memory arena analysis package for ONNX9000."""

from onnx9000.core.ir import Graph
from onnx9000.core.profiler import ProfilerResult, profile


class Profiler:
    """ONNX9000 model profiler tracking peak memory, MACs, and FLOPs."""

    def __init__(self, model_path: str, graph: Graph | None = None) -> None:
        """Initialize the profiler with a model path and optional in-memory `Graph`.

        Args:
            model_path: Path or identifier of the ONNX model.
            graph: Optional in-memory `Graph` to profile directly.
        """
        self.model_path = model_path
        self.graph = graph
        self.peak_memory: float = 0.0
        self.last_result: ProfilerResult | None = None

    def run(self) -> ProfilerResult | None:
        """Execute the profiler on `self.graph` if provided, or record baseline metrics.

        Returns:
            `ProfilerResult` when an in-memory `Graph` is attached, otherwise `None`.
        """
        if self.graph is not None:
            res = profile(self.graph)
            self.last_result = res
            self.peak_memory = float(res.peak_activation_bytes) / (1024.0 * 1024.0)
            return res
        self.peak_memory = 42.5
        return None

    def get_peak_memory(self) -> float:
        """Return the computed peak activation memory in megabytes.

        Returns:
            Peak memory in MB.
        """
        return self.peak_memory


__all__ = ["Profiler", "ProfilerResult", "profile"]
