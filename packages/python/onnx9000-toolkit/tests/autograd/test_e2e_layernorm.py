from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.toolkit.training.autograd.compiler import AOTBuilder
from onnx9000.toolkit.training.autograd.losses import add_mse_loss


def test_train_layernorm() -> None:
    g = Graph("ln")
    g.inputs.extend(["x", "target", "lr"])
    g.add_tensor(Tensor("x", shape=(1, 16, 64), dtype="float32", requires_grad=True))
    g.add_tensor(Tensor("target", shape=(1, 16, 64), dtype="float32"))
    g.add_tensor(Tensor("lr", shape=(), dtype="float32"))

    g.initializers.extend(["gamma", "beta"])
    g.add_tensor(Tensor("gamma", shape=(64,), dtype="float32", requires_grad=True))
    g.add_tensor(Tensor("beta", shape=(64,), dtype="float32", requires_grad=True))

    g.add_node(Node("LayerNormalization", ["x", "gamma", "beta"], ["pred"], {"axis": -1}))

    g.add_tensor(Tensor("pred", shape=(1, 16, 64), dtype="float32"))
    g.outputs.append("pred")

    builder = AOTBuilder(g)

    def loss_gen(gr, p, t, o):
        add_mse_loss(gr, p, t, o)

    def opt_gen(gr, lr, p):
        pass

    train_graph = builder.build_training_graph(loss_gen, opt_gen, "lr")
    assert train_graph is not None
