from onnx9000.core.ir import Node
from onnx9000.toolkit.training.autograd.vjp import VJPRule


class MockRule(VJPRule):
    def build_backward_nodes(self, fwd_node, grad_outputs):
        return super().build_backward_nodes(fwd_node, grad_outputs)


def test_vjp_base() -> None:
    r = MockRule()
    res = r.build_backward_nodes(Node("Test", [], [], {}), [])
    assert res == ([], [])
