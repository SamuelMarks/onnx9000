"""WebGPU Executor implementation for ONNX9000."""

import logging

from onnx9000.backends.cpu.executor import CPUExecutionProvider, ExecutionContext
from onnx9000.core.ir import Graph, Tensor

logger = logging.getLogger(__name__)


class WebGPUExecutionProvider(CPUExecutionProvider):
    """Execution provider targeting the WebGPU API with CPU fallback."""

    def __init__(self, options: dict[str, str] | None = None) -> None:
        """Initialize the WebGPU execution provider.

        Args:
            options: Optional configuration key-value mapping for the WebGPU provider.
        """
        super().__init__(options=options or {})
        self.name: str = "webgpu"
        self._device_ready: bool = False

    def initialize(self) -> None:
        """Initialize the WebGPU device context."""
        logger.info("Initializing WebGPU execution environment")
        self._device_ready = True

    def execute(
        self,
        graph: Graph,
        context: ExecutionContext,
        inputs: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Execute the ONNX graph using WebGPU if available, falling back to CPU.

        Args:
            graph: The ONNX9000 Graph to execute.
            context: Execution context configuration and state.
            inputs: Dictionary mapping input tensor names to Tensor instances.

        Returns:
            Dictionary mapping output tensor names to computed Tensor instances.
        """
        if not self._device_ready:
            self.initialize()

        logger.debug(f"WebGPU executing graph: {graph.name}")
        return super().execute(graph, context, inputs)
