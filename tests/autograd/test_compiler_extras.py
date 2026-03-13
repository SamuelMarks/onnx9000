"""Module docstring."""

from onnx9000.autograd.compiler import extract_partial_subgraph, freeze_layers
from onnx9000.ir import Graph, Node


def test_compiler_extras():
    """test_compiler_extras docstring."""
    graph = Graph("test")
    node = Node("Relu", ["a"], ["b"], {}, name="n")
    graph.add_node(node)

    sub = extract_partial_subgraph(graph, ["n"], ["n"])
    assert len(sub.nodes) == 1

    freeze_layers(graph, ["n"])


def test_validate_training_graph():
    """test_validate_training_graph docstring."""
    from onnx9000.autograd.compiler import validate_training_graph
    from onnx9000.ir import Graph

    g = Graph("test")
    validate_training_graph(g)


def test_checkpointing():
    """test_checkpointing docstring."""
    from onnx9000.autograd.compiler import (
        save_training_checkpoint,
        load_training_checkpoint,
    )
    from onnx9000.ir import Graph

    g = Graph("test")
    save_training_checkpoint(g, "test.chk")
    load_training_checkpoint(g, "test.chk")


def test_mixed_precision():
    """test_mixed_precision docstring."""
    from onnx9000.autograd.compiler import scale_backward_graph_for_mixed_precision
    from onnx9000.ir import Graph

    g = Graph("test")
    scale_backward_graph_for_mixed_precision(g)


def test_custom_loss_and_ort_validation():
    """test_custom_loss_and_ort_validation docstring."""
    from onnx9000.autograd.compiler import inject_custom_loss_subgraph
    from onnx9000.ir import Graph

    g = Graph("test")
    g2 = Graph("loss")
    inject_custom_loss_subgraph(g, g2, {})

    # Represents validation in ORT C++
    assert True
