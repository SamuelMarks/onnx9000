from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.toolkit.training.autograd.compiler import AOTBuilder
from onnx9000.toolkit.training.autograd.losses import add_mse_loss
from onnx9000.toolkit.training.autograd.optimizers import add_sgd_optimizer


def test_train_mlp() -> None:
    g = Graph("mlp")
    g.inputs.append("x")
    g.inputs.append("target")
    g.inputs.append("lr")
    g.add_tensor(Tensor("x", shape=(1, 4), dtype="float32"))
    g.add_tensor(Tensor("target", shape=(1, 2), dtype="float32"))
    g.add_tensor(Tensor("lr", shape=(), dtype="float32"))

    g.initializers.extend(["w1", "b1", "w2", "b2"])
    g.add_tensor(Tensor("w1", shape=(4, 8), dtype="float32", requires_grad=True))
    g.add_tensor(Tensor("b1", shape=(8,), dtype="float32", requires_grad=True))
    g.add_tensor(Tensor("w2", shape=(8, 2), dtype="float32", requires_grad=True))
    g.add_tensor(Tensor("b2", shape=(2,), dtype="float32", requires_grad=True))

    g.add_node(Node("MatMul", ["x", "w1"], ["xw1"]))
    g.add_node(Node("Add", ["xw1", "b1"], ["h1"]))
    g.add_node(Node("Relu", ["h1"], ["a1"]))
    g.add_node(Node("MatMul", ["a1", "w2"], ["xw2"]))
    g.add_node(Node("Add", ["xw2", "b2"], ["pred"]))

    g.add_tensor(Tensor("pred", shape=(1, 2), dtype="float32"))
    g.outputs.append("pred")

    builder = AOTBuilder(g)

    def loss_gen(gr, p, t, o):
        add_mse_loss(gr, p, t, o)

    def opt_gen(gr, lr, p):
        add_sgd_optimizer(gr, lr, p)

    train_graph = builder.build_training_graph(loss_gen, opt_gen, "lr")
    assert train_graph is not None
