"""Native Execution Orchestrator."""

import os
import time
import logging
import numpy as np
from typing import Any, Optional

from onnx9000.core.ir import Graph, Node
from onnx9000.backends.cpu.executor import Executor as CPUExecutor
from onnx9000.backends.cuda.executor import Dispatcher as CUDADispatcher
from onnx9000.backends.cuda.bindings import is_cuda_available
from onnx9000.backends.rocm.executor import Dispatcher as ROCmDispatcher
from onnx9000.backends.rocm.bindings import is_hip_available
from onnx9000.backends.apple.executor import Dispatcher as AppleDispatcher
from onnx9000.backends.apple.bindings import is_metal_available, is_accelerate_available

logger = logging.getLogger(__name__)


class NativeSessionOptions:
    """Represents the NativeSessionOptions class."""

    def __init__(self):
        """Provides   init   functionality and verification."""
        self.enable_profiling = False
        self.device = "auto"
        self.enable_mem_pattern = True
        self.enable_cpu_mem_arena = True
        self.log_severity_level = 2
        self.cache_dir = os.path.join(os.path.expanduser("~"), ".onnx9000", "cache")


class NativeSession:
    """Mirrors ORT InferenceSession API."""

    def __init__(
        self,
        path_or_bytes: Any,
        sess_options: Optional[NativeSessionOptions] = None,
        providers: Optional[list[str]] = None,
    ) -> None:
        """Provides   init   functionality and verification."""
        self.options = sess_options or NativeSessionOptions()
        if not os.path.exists(self.options.cache_dir):
            try:
                os.makedirs(self.options.cache_dir, exist_ok=True)
            except Exception as e:
                logger.debug(f"Failed to create cache dir: {e}")

        self.providers = providers or ["CPUExecutionProvider"]
        self.graph = (
            path_or_bytes if isinstance(path_or_bytes, Graph) else Graph("empty")
        )

        self.device = self._auto_detect_device()
        self.executor = self._route_graph_to_dispatcher()
        self.profiling_data: list[dict[str, Any]] = []

    def _auto_detect_device(self) -> str:
        """Hardware auto-detection."""
        if self.options.device != "auto":
            return self.options.device

        if is_cuda_available():
            return "cuda"
        if is_hip_available():
            return "rocm"
        if is_metal_available() or is_accelerate_available():
            return "metal"
        return "cpu"

    def _route_graph_to_dispatcher(self) -> Any:
        """Automatically route the ir.Graph to the optimal Dispatcher."""
        if self.device.startswith("cuda"):
            return CUDADispatcher(self.graph)
        elif self.device.startswith("rocm"):
            return ROCmDispatcher(self.graph)
        elif self.device == "metal":
            return AppleDispatcher(self.graph)
        else:
            return CPUExecutor(self.graph)

    def run(
        self, output_names: Optional[list[str]], input_feed: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        """Executes the run operation."""
        start_time = time.time()

        if self.options.enable_profiling:
            logger.info("Profiling enabled. Starting run...")

        results_dict = self.executor.run(input_feed)

        end_time = time.time()
        latency = end_time - start_time

        if self.options.enable_profiling:
            self.profiling_data.append(
                {"event": "run", "latency_ms": latency * 1000.0, "device": self.device}
            )

        if output_names is None:
            output_names = self.graph.outputs

        return [results_dict.get(name, np.array([])) for name in output_names]

    def get_profiling_start_time_ns(self) -> int:
        """Executes the get profiling start time ns operation."""
        return int(time.time() * 1e9)

    def end_profiling(self) -> str:
        """Executes the end profiling operation."""
        if not self.options.enable_profiling:
            return ""

        path = os.path.join(self.options.cache_dir, f"profile_{int(time.time())}.json")
        try:
            import json

            with open(path, "w") as f:
                json.dump(self.profiling_data, f)
        except Exception as e:
            logger.debug(f"Failed to dump profiling data: {e}")
        return path

    def partition_graph(self) -> tuple[Graph, Graph]:
        """Implement a graph partitioning pass to separate CPU and GPU execution streams optimally."""
        cpu_graph = Graph(f"{self.graph.name}_cpu")
        gpu_graph = Graph(f"{self.graph.name}_gpu")

        # Simple heuristic: elementwise ops on CPU, matmuls/convs on GPU
        for node in self.graph.nodes:
            if node.op_type in ["Add", "Sub", "Mul", "Div"]:
                cpu_graph.add_node(node)
            else:
                gpu_graph.add_node(node)

        return cpu_graph, gpu_graph

    def profile_memory_utilization(self) -> dict[str, float]:
        """Profile memory utilization using the planner's actual state."""
        arena_size = 0.0
        if hasattr(self.executor, "planner") and hasattr(
            self.executor.planner, "current_offset"
        ):
            arena_size = float(self.executor.planner.current_offset)

        return {"arena_bytes": arena_size, "overhead_bytes": 0.0}
