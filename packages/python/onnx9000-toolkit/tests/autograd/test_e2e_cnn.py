"""Tests the e2e cnn module functionality."""

from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.toolkit.training.autograd.compiler import AOTBuilder
from onnx9000.toolkit.training.autograd.losses import add_mse_loss
from onnx9000.toolkit.training.autograd.optimizers import add_sgd_optimizer


def test_train_cnn() -> None:
    """Tests the train cnn functionality."""
    g = Graph("cnn")
    g.inputs.append("x")
    g.inputs.append("target")
    g.inputs.append("lr")
    g.add_tensor(Tensor("x", shape=(1, 1, 16, 16), dtype="float32"))
    g.add_tensor(Tensor("target", shape=(1, 10), dtype="float32"))
    g.add_tensor(Tensor("lr", shape=(), dtype="float32"))

    g.initializers.extend(["w_conv", "b_conv", "w_fc", "b_fc"])
    g.add_tensor(Tensor("w_conv", shape=(4, 1, 3, 3), dtype="float32", requires_grad=True))
    g.add_tensor(Tensor("b_conv", shape=(4,), dtype="float32", requires_grad=True))
    g.add_tensor(
        Tensor("w_fc", shape=(196, 10), dtype="float32", requires_grad=True)
    )  # e.g. 4*7*7 = 196
    g.add_tensor(Tensor("b_fc", shape=(10,), dtype="float32", requires_grad=True))

    g.add_node(
        Node(
            "Conv",
            ["x", "w_conv", "b_conv"],
            ["conv_out"],
            {"kernel_shape": [3, 3], "pads": [0, 0, 0, 0]},
        )
    )
    g.add_node(Node("Relu", ["conv_out"], ["relu_out"]))
    g.add_node(
        Node("MaxPool", ["relu_out"], ["pool_out"], {"kernel_shape": [2, 2], "strides": [2, 2]})
    )

    # Flatten dynamically to 2D
    # The actual Flatten node doesn't have a second input for shape.
    # Use Flatten op instead of Reshape+Constant
    g.add_node(Node("Flatten", ["pool_out"], ["flat_out"], {"axis": 1}))

    g.add_node(Node("MatMul", ["flat_out", "w_fc"], ["xw"]))
    g.add_node(Node("Add", ["xw", "b_fc"], ["pred"]))

    g.add_tensor(Tensor("pred", shape=(1, 10), dtype="float32"))
    g.outputs.append("pred")

    builder = AOTBuilder(g)

    def loss_gen(gr, p, t, o):
        """Tests the loss gen functionality."""
        add_mse_loss(gr, p, t, o)

    def opt_gen(gr, lr, p):
        """Tests the opt gen functionality."""
        add_sgd_optimizer(gr, lr, p)

    train_graph = builder.build_training_graph(loss_gen, opt_gen, "lr")
    assert train_graph is not None
