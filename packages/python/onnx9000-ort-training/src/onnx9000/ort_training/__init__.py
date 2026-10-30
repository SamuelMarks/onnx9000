"""Ahead-of-Time (AOT) autograd and training graph builder (`ORTTraining`) for ONNX9000."""

from onnx9000.core.ir import Graph
from onnx9000.toolkit.training import AOTBuilder, compile_training_graph


class ORTTraining:
    """High-level controller for compiling forward ONNX graphs into AOT training graphs."""

    def create_builder(self, graph: Graph) -> AOTBuilder:
        """Create an `AOTBuilder` for the given forward `Graph`.

        Args:
            graph: Forward-pass ONNX9000 `Graph`.

        Returns:
            Initialized `AOTBuilder`.
        """
        return AOTBuilder(graph)

    def process(self, arg: str) -> str:
        """Process a training job descriptor string.

        Args:
            arg: Training job descriptor.

        Returns:
            Formatted ORT Training status string.
        """
        return f"ORT Training processed {arg}"


__all__ = ["AOTBuilder", "ORTTraining", "compile_training_graph"]
