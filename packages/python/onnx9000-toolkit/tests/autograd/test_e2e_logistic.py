"""Tests the e2e logistic module functionality."""

from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.toolkit.training.autograd.compiler import AOTBuilder
from onnx9000.toolkit.training.autograd.losses import add_bce_with_logits_loss
from onnx9000.toolkit.training.autograd.optimizers import add_sgd_optimizer


def test_train_logistic_regression() -> None:
    """Tests the train logistic regression functionality."""
    g = Graph("logistic")
    g.inputs.append("x")
    g.inputs.append("target")
    g.inputs.append("lr")
    g.add_tensor(Tensor("x", shape=(1, 2), dtype="float32"))
    g.add_tensor(Tensor("target", shape=(1, 1), dtype="float32"))
    g.add_tensor(Tensor("lr", shape=(), dtype="float32"))

    g.initializers.append("w")
    g.initializers.append("b")
    g.add_tensor(Tensor("w", shape=(2, 1), dtype="float32", requires_grad=True))
    g.add_tensor(Tensor("b", shape=(1,), dtype="float32", requires_grad=True))

    g.add_node(Node("MatMul", ["x", "w"], ["xw"]))
    g.add_node(Node("Add", ["xw", "b"], ["logits"]))
    g.add_tensor(Tensor("logits", shape=(1, 1), dtype="float32"))
    g.outputs.append("logits")

    builder = AOTBuilder(g)

    def loss_gen(gr, p, t, o):
        """Test the loss gen functionality."""
        add_bce_with_logits_loss(gr, p, t, o)

    def opt_gen(gr, lr, p):
        """Test the opt gen functionality."""
        add_sgd_optimizer(gr, lr, p)

    train_graph = builder.build_training_graph(loss_gen, opt_gen, "lr")
    assert train_graph is not None
