"""Module providing core logic and structural definitions."""

from onnx9000.training.autograd.compiler import extract_partial_subgraph, freeze_layers
from onnx9000.core.ir import Graph, Node


def test_compiler_extras():
    """Tests the test compiler extras functionality."""
    graph = Graph("test")
    node = Node("Relu", ["a"], ["b"], {}, name="n")
    graph.add_node(node)
    from onnx9000.core.ir import Tensor

    graph.add_tensor(Tensor("dummy_tensor", (), "float32"))
    sub = extract_partial_subgraph(graph, ["n"], ["n"])
    assert len(sub.nodes) == 1
    freeze_layers(graph, ["n"])


def test_validate_training_graph():
    """Tests the test validate training graph functionality."""
    from onnx9000.training.autograd.compiler import validate_training_graph
    from onnx9000.core.ir import Graph

    g = Graph("test")
    validate_training_graph(g)


def test_checkpointing():
    """Tests the test checkpointing functionality."""
    from onnx9000.training.autograd.compiler import (
        save_training_checkpoint,
        load_training_checkpoint,
    )
    from onnx9000.core.ir import Graph

    g = Graph("test")
    save_training_checkpoint(g, "test.chk")
    load_training_checkpoint(g, "test.chk")


def test_mixed_precision():
    """Tests the test mixed precision functionality."""
    from onnx9000.training.autograd.compiler import (
        scale_backward_graph_for_mixed_precision,
    )
    from onnx9000.core.ir import Graph

    g = Graph("test")
    scale_backward_graph_for_mixed_precision(g)


def test_custom_loss_and_ort_validation():
    """Tests the test custom loss and ort validation functionality."""
    from onnx9000.training.autograd.compiler import inject_custom_loss_subgraph
    from onnx9000.core.ir import Graph

    g = Graph("test")
    g2 = Graph("loss")
    inject_custom_loss_subgraph(g, g2, {})
    assert len(g.nodes) == 0
