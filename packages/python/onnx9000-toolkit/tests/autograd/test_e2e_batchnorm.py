"""Tests the e2e batchnorm module functionality."""

from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.toolkit.training.autograd.compiler import AOTBuilder
from onnx9000.toolkit.training.autograd.losses import add_mse_loss


def test_train_batchnorm() -> None:
    """Tests the train batchnorm functionality."""
    g = Graph("bn")
    g.inputs.extend(["x", "target", "lr"])
    g.add_tensor(Tensor("x", shape=(1, 64, 14, 14), dtype="float32", requires_grad=True))
    g.add_tensor(Tensor("target", shape=(1, 64, 14, 14), dtype="float32"))
    g.add_tensor(Tensor("lr", shape=(), dtype="float32"))

    g.initializers.extend(["gamma", "beta", "mean", "var"])
    g.add_tensor(Tensor("gamma", shape=(64,), dtype="float32", requires_grad=True))
    g.add_tensor(Tensor("beta", shape=(64,), dtype="float32", requires_grad=True))
    g.add_tensor(Tensor("mean", shape=(64,), dtype="float32", requires_grad=False))
    g.add_tensor(Tensor("var", shape=(64,), dtype="float32", requires_grad=False))

    g.add_node(
        Node(
            "BatchNormalization",
            ["x", "gamma", "beta", "mean", "var"],
            ["pred"],
            {"training_mode": 1},
        )
    )

    g.add_tensor(Tensor("pred", shape=(1, 64, 14, 14), dtype="float32"))
    g.outputs.append("pred")

    builder = AOTBuilder(g)

    def loss_gen(gr, p, t, o):
        """Tests the loss gen functionality."""
        add_mse_loss(gr, p, t, o)

    def opt_gen(gr, lr, p):
        """Tests the opt gen functionality."""
        pass

    train_graph = builder.build_training_graph(loss_gen, opt_gen, "lr")
    assert train_graph is not None
