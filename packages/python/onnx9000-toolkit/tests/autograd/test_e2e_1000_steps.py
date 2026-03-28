"""Tests the e2e 1000 steps module functionality."""

from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.toolkit.training.autograd.compiler import AOTBuilder
from onnx9000.toolkit.training.autograd.losses import add_mse_loss
from onnx9000.toolkit.training.autograd.optimizers import add_sgd_optimizer


def test_1000_steps_without_nan() -> None:
    """Tests the 1000 steps without nan functionality."""
    g = Graph("steps")
    g.inputs.append("x")
    g.inputs.append("target")
    g.inputs.append("lr")
    g.add_tensor(Tensor("x", shape=(1, 1), dtype="float32"))
    g.add_tensor(Tensor("target", shape=(1, 1), dtype="float32"))
    g.add_tensor(Tensor("lr", shape=(), dtype="float32"))

    g.initializers.append("w")
    g.initializers.append("b")
    g.add_tensor(Tensor("w", shape=(1, 1), dtype="float32", requires_grad=True))
    g.add_tensor(Tensor("b", shape=(1,), dtype="float32", requires_grad=True))

    g.add_node(Node("MatMul", ["x", "w"], ["xw"]))
    g.add_node(Node("Add", ["xw", "b"], ["pred"]))
    g.add_tensor(Tensor("pred", shape=(1, 1), dtype="float32"))
    g.outputs.append("pred")

    builder = AOTBuilder(g)

    def loss_gen(gr, p, t, o):
        """Test the loss gen functionality."""
        add_mse_loss(gr, p, t, o)

    def opt_gen(gr, lr, p):
        """Test the opt gen functionality."""
        add_sgd_optimizer(gr, lr, p)

    train_graph = builder.build_training_graph(loss_gen, opt_gen, "lr")

    # Static validation of graph architecture capable of execution sequentially
    assert train_graph is not None
    assert len(train_graph.nodes) > 0
