from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.toolkit.training.autograd.compiler import build_backward_graph
import pytest


def test_custom_domain_raise():
    g = Graph("test")
    g.inputs.append("in")
    g.add_tensor(Tensor(name="in", shape=(), dtype="float32", requires_grad=True))
    g.add_tensor(Tensor(name="out", shape=(), dtype="float32"))
    n = Node("MyCustomOp", ["in"], ["out"], domain="com.example")
    g.add_node(n)
    g.outputs.append("out")

    with pytest.raises(RuntimeError) as exc:
        build_backward_graph(g)
    assert "belongs to custom domain" in str(exc.value)
