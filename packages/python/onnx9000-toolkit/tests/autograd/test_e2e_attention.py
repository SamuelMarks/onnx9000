"""Tests the e2e attention module functionality."""

from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.toolkit.training.autograd.compiler import AOTBuilder
from onnx9000.toolkit.training.autograd.losses import add_mse_loss


def test_train_transformer_attention() -> None:
    """Tests the train transformer attention functionality."""
    g = Graph("attention")
    g.inputs.extend(["q", "k", "v", "target", "lr"])
    g.add_tensor(
        Tensor("q", shape=(1, 8, 16, 64), dtype="float32", requires_grad=True)
    )  # batch, heads, seq, dim
    g.add_tensor(Tensor("k", shape=(1, 8, 16, 64), dtype="float32", requires_grad=True))
    g.add_tensor(Tensor("v", shape=(1, 8, 16, 64), dtype="float32", requires_grad=True))
    g.add_tensor(Tensor("target", shape=(1, 8, 16, 64), dtype="float32"))
    g.add_tensor(Tensor("lr", shape=(), dtype="float32"))

    # K transposed
    g.add_node(Node("Transpose", ["k"], ["k_t"], {"perm": [0, 1, 3, 2]}))

    # Q * K^T
    g.add_node(Node("MatMul", ["q", "k_t"], ["scores"]))

    # Softmax
    g.add_node(Node("Softmax", ["scores"], ["attn_weights"], {"axis": -1}))

    # V * weights
    g.add_node(Node("MatMul", ["attn_weights", "v"], ["pred"]))

    g.add_tensor(Tensor("pred", shape=(1, 8, 16, 64), dtype="float32"))
    g.outputs.append("pred")

    # No initializers, testing input gradients tracking
    builder = AOTBuilder(g)

    def loss_gen(gr, p, t, o):
        """Test the loss gen functionality."""
        add_mse_loss(gr, p, t, o)

    # Pass empty params to optimizer since QKV are inputs, not initializers
    def opt_gen(gr, lr, p):
        """Test the opt gen functionality."""
        return None

    train_graph = builder.build_training_graph(loss_gen, opt_gen, "lr")
    assert train_graph is not None
