"""Module providing core logic and structural definitions."""

from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.toolkit.training.autograd.compiler import (
    AOTBuilder,
    extract_partial_subgraph,
    freeze_layers,
    inject_custom_loss_subgraph,
    load_training_checkpoint,
    save_training_checkpoint,
    scale_backward_graph_for_mixed_precision,
    validate_training_graph,
)
from onnx9000.toolkit.training.autograd.losses import add_mse_loss
from onnx9000.toolkit.training.autograd.optimizers import add_sgd_optimizer


def test_aot_builder() -> None:
    """Tests the test_aot_builder functionality."""
    g = Graph("test")
    g.inputs.append("x")
    g.outputs.append("y")
    g.initializers.append("w")
    g.add_tensor(Tensor("x", (10,), "float32"))
    g.add_tensor(Tensor("w", (10,), "float32", is_initializer=True))
    g.add_tensor(Tensor("y", (10,), "float32"))
    g.add_node(Node("Add", ["x", "w"], ["y"], {}))
    builder = AOTBuilder(g)
    train_g = builder.build_training_graph(add_mse_loss, add_sgd_optimizer, "lr")
    assert train_g.name == "test_aot_training_training"
    assert "loss" in train_g.outputs
    assert "target" in train_g.inputs
    assert "lr" in train_g.inputs


def test_extras(tmp_path) -> None:
    """Tests the test_extras functionality."""
    g = Graph("test")
    sub = extract_partial_subgraph(g, [], [])
    assert sub.name == "test_partial"
    validate_training_graph(g)
    cp = tmp_path / "cp.txt"
    save_training_checkpoint(g, str(cp))
    assert cp.exists()
    load_training_checkpoint(g, str(cp))
    freeze_layers(g, [])
    scale_backward_graph_for_mixed_precision(g)
    g2 = Graph("loss")
    g2.add_node(Node("Sub", [], [], {}))
    inject_custom_loss_subgraph(g, g2, {})
    assert len(g.nodes) == 1
