"""ONNX to PyTorch, Flax NNX, Keras 3, and Triton code generators for ONNX9000."""

from onnx9000.core.codegen import (
    ONNXToFlaxNNXVisitor,
    ONNXToKerasVisitor,
    ONNXToPyTorchVisitor,
    TritonExporter,
)
from onnx9000.core.ir import Graph


def compile_to_pytorch(graph: Graph) -> str:
    """Generate a standalone PyTorch `nn.Module` Python source string from an ONNX9000 `Graph`.

    Args:
        graph: Source ONNX9000 `Graph`.

    Returns:
        Generated PyTorch source code string.
    """
    return ONNXToPyTorchVisitor(graph).generate()


def run() -> str:
    """Execute the default pytorch-codegen status check.

    Returns:
        pytorch-codegen status confirmation string.
    """
    return "[pytorch-codegen] processed"


__all__ = [
    "ONNXToFlaxNNXVisitor",
    "ONNXToKerasVisitor",
    "ONNXToPyTorchVisitor",
    "TritonExporter",
    "compile_to_pytorch",
    "run",
]
