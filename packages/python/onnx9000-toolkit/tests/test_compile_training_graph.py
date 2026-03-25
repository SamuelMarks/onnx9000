import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.toolkit.training import compile_training_graph


def test_compile_training_graph():
    g = Graph("TestCompile")
    g.inputs = ["X"]
    g.outputs = ["Y"]
    g.tensors["X"] = Tensor("X", (2, 2), DType.FLOAT32)
    g.tensors["W"] = Tensor("W", (2, 2), DType.FLOAT32, is_initializer=True)
    g.tensors["Y"] = Tensor("Y", (2, 2), DType.FLOAT32)

    n = Node("MatMul", inputs=["X", "W"], outputs=["Y"])
    g.nodes.append(n)

    # Mock loss and optim
    def mock_loss(out, target, b):
        return "loss_out"

    mock_loss(None, None, None)

    def mock_optim(builder, params, grads, lr):
        pass

    mock_optim(None, None, None, None)

    # Note: we need builder to actually not crash. AOTBuilder calls these.
    # It might be easier to use real ones from the toolkit if possible, or mock AOTBuilder.
