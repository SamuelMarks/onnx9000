"""Tests the test vjp base functionality."""

from onnx9000.training.autograd.vjp import VJPRule
from onnx9000.core.ir import Node


class DummyRule(VJPRule):
    """Represents the DummyRule class."""

    def build_backward_nodes(self, fwd_node, grad_outputs):
        """Provides build backward nodes functionality and verification."""
        return super().build_backward_nodes(fwd_node, grad_outputs)


def test_vjp_rule_base():
    """Tests the test vjp rule base functionality."""
    rule = DummyRule()
    node = Node("Relu", ["a"], ["b"], {})
    nodes, names = rule.build_backward_nodes(node, ["grad_b"])
    assert nodes == []
    assert names == []
