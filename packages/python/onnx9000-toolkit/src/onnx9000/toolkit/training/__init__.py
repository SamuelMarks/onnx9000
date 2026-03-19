"""Module containing training API endpoints."""

from onnx9000.core.ir import Graph
from onnx9000.toolkit.training.autograd.compiler import AOTBuilder


def compile_training_graph(
    model: Graph, loss_fn, optimizer_fn, learning_rate: str = "learning_rate"
) -> Graph:
    """Expose Python API for compiling the training graph.

    Compiles the provided ONNX IR Graph into a monolithic training graph containing
    forward pass, loss, backward pass, and optimizer steps.
    """
    builder = AOTBuilder(model)
    return builder.build_training_graph(loss_fn, optimizer_fn, learning_rate)


__all__ = ["compile_training_graph", "AOTBuilder"]
