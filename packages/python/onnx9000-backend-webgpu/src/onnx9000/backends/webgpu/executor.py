"""WebGPU Executor implementation."""

import logging
from typing import Any, Optional

from onnx9000.backends.cpu.executor import CPUExecutionProvider
from onnx9000.core.ir import Graph, Tensor

logger = logging.getLogger(__name__)


class WebGPUExecutionProvider(CPUExecutionProvider):
    """
    Execution provider targeting WebGPU API.
    Provides fallback to CPU for unsupported operations.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the WebGPU execution provider."""
        super().__init__(options={}, **kwargs)
        self.name = "webgpu"
        self._device_ready = False

    def initialize(self) -> None:
        """Initialize the WebGPU device context."""
        logger.info("Initializing WebGPU execution environment (stub)")
        self._device_ready = True
        pass  # CPUProvider might not have initialize

    def execute(self, graph: Graph, context: Any, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Execute the ONNX graph using WebGPU if available, falling back to CPU.
        """
        if not self._device_ready:
            self.initialize()

        logger.debug(f"WebGPU executing graph: {graph.name}")
        # In a full implementation, this would dispatch to WGSL shaders
        # For now, we fallback to CPU implementation
        return super().execute(graph, context, inputs)
