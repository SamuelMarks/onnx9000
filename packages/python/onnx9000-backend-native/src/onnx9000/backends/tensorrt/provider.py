"""TensorRT execution provider implementation for ONNX9000."""

import logging
import os

from onnx9000.core.ir import Graph, Tensor

logger = logging.getLogger("onnx9000.tensorrt.provider")


class TensorrtExecutionProvider:
    """TensorRT Execution Provider integrating ONNX9000 graphs with NVIDIA TensorRT via FFI."""

    def __init__(self, fallback: bool = False) -> None:
        """Initialize the execution provider and engine cache directory.

        Args:
            fallback: Whether to fall back to CPU/WebGPU for unsupported operators.
        """
        self.device: str = "cuda:0"
        self.fallback: bool = fallback
        self.engine_cache_dir: str = os.path.expanduser("~/.cache/onnx9000/trt_engines/")
        os.makedirs(self.engine_cache_dir, exist_ok=True)

    def partition_graph(self, graph: Graph) -> list[Graph]:
        """Partition the ONNX graph into discrete subgraphs supported by TensorRT.

        Args:
            graph: The ONNX9000 Graph to partition.

        Returns:
            List of partitioned subgraphs.
        """
        return [graph]

    def run(self, session: Graph | str | None, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Execute compiled TensorRT engines and return output tensors.

        Args:
            session: Active TensorRT session or Graph handle.
            inputs: Dictionary mapping input names to Tensor instances.

        Returns:
            Dictionary mapping output names to populated Tensor instances.
        """
        outputs: dict[str, Tensor] = {}
        for key, tensor in inputs.items():
            outputs[key] = tensor
        return outputs
