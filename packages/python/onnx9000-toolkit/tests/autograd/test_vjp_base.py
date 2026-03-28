"""Tests the vjp base module functionality."""

from onnx9000.core.ir import Node
from onnx9000.toolkit.training.autograd.vjp import VJPRule


class MockRule(VJPRule):
    """Represents the Mock Rule class."""

    def build_backward_nodes(self, fwd_node, grad_outputs):
        """Execute the build backward nodes operation."""
        return super().build_backward_nodes(fwd_node, grad_outputs)


def test_vjp_base() -> None:
    """Tests the vjp base functionality."""
    r = MockRule()
    res = r.build_backward_nodes(Node("Test", [], [], {}), [])
    assert res == ([], [])
