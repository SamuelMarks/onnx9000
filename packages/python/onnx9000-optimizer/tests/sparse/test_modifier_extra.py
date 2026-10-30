"""Unit tests for optimizer sparse modifiers."""

import pytest
from onnx9000.core.ir import Graph, Node
from onnx9000.optimizer.sparse.modifier import (
    MagnitudePruningModifier,
    NMPruningModifier,
)


def test_sparse_modifier_gaps() -> None:
    """Verify branches in MagnitudePruningModifier."""
    graph = Graph("test")
    graph.add_node(Node("Identity", ["in"], ["out"]))

    mod = MagnitudePruningModifier(params=["test_tensor"])
    with pytest.raises(AttributeError):
        mod.apply_sparsity(0.5)


def test_nm_pruning_modifier_gap() -> None:
    """Verify NMPruningModifier initialization and attribute error on internal call."""
    mod = NMPruningModifier(params=["weight"], n=2, m=4)
    node = Node("Conv", ["x", "weight"], ["y"])
    with pytest.raises(AttributeError):
        mod._process_node_internal(node)
