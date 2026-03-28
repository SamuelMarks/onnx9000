"""Tests the e2e resnet module functionality."""

from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.toolkit.training.autograd.compiler import AOTBuilder
from onnx9000.toolkit.training.autograd.losses import add_mse_loss
from onnx9000.toolkit.training.autograd.optimizers import add_sgd_optimizer


def test_train_resnet_bottleneck() -> None:
    """Tests the train resnet bottleneck functionality."""
    g = Graph("resnet_bottleneck")
    g.inputs.append("x")
    g.inputs.append("target")
    g.inputs.append("lr")

    g.add_tensor(Tensor("x", shape=(1, 256, 14, 14), dtype="float32"))
    g.add_tensor(Tensor("target", shape=(1, 256, 14, 14), dtype="float32"))
    g.add_tensor(Tensor("lr", shape=(), dtype="float32"))

    # Weights for three conv layers in a standard bottleneck
    g.initializers.extend(["w1", "b1", "w2", "b2", "w3", "b3"])
    g.add_tensor(Tensor("w1", shape=(64, 256, 1, 1), dtype="float32", requires_grad=True))
    g.add_tensor(Tensor("b1", shape=(64,), dtype="float32", requires_grad=True))
    g.add_tensor(Tensor("w2", shape=(64, 64, 3, 3), dtype="float32", requires_grad=True))
    g.add_tensor(Tensor("b2", shape=(64,), dtype="float32", requires_grad=True))
    g.add_tensor(Tensor("w3", shape=(256, 64, 1, 1), dtype="float32", requires_grad=True))
    g.add_tensor(Tensor("b3", shape=(256,), dtype="float32", requires_grad=True))

    # Main path
    g.add_node(
        Node("Conv", ["x", "w1", "b1"], ["c1"], {"kernel_shape": [1, 1], "pads": [0, 0, 0, 0]})
    )
    g.add_node(Node("Relu", ["c1"], ["r1"]))

    g.add_node(
        Node("Conv", ["r1", "w2", "b2"], ["c2"], {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1]})
    )
    g.add_node(Node("Relu", ["c2"], ["r2"]))

    g.add_node(
        Node("Conv", ["r2", "w3", "b3"], ["c3"], {"kernel_shape": [1, 1], "pads": [0, 0, 0, 0]})
    )

    # Residual addition
    g.add_node(Node("Add", ["c3", "x"], ["pred"]))
    g.add_tensor(Tensor("pred", shape=(1, 256, 14, 14), dtype="float32"))
    g.outputs.append("pred")

    builder = AOTBuilder(g)

    def loss_gen(gr, p, t, o):
        """Test the loss gen functionality."""
        add_mse_loss(gr, p, t, o)

    def opt_gen(gr, lr, p):
        """Test the opt gen functionality."""
        add_sgd_optimizer(gr, lr, p)

    train_graph = builder.build_training_graph(loss_gen, opt_gen, "lr")
    assert train_graph is not None
