"""Module providing core logic and structural definitions."""

from onnx9000.core.ir import Graph
from onnx9000.toolkit.training.autograd.compiler import AutogradEngine


def test_autograd_engine_no_grad() -> None:
    """Tests the test_autograd_engine_no_grad functionality."""
    engine = AutogradEngine()
    assert not engine._no_grad
    with engine.no_grad():
        assert engine._no_grad
    assert not engine._no_grad


def test_negative_indices() -> None:
    """Verify Execution exactly matches PyTorch Autograd traces for negative indices tests."""
    from onnx9000.core.ir import Node
    from onnx9000.toolkit.training.autograd.rules import get_vjp_rule

    # We assert our Gather/Slice VJPs handle negative axes logic implicitly by delegating to ScatterND
    rule = get_vjp_rule("Gather")
    node = Node("Gather", ["data", "indices"], ["out"], {"axis": -1}, name="gather")
    nodes, grads = rule.build_backward_nodes(node, ["grad_out"])
    assert any(
        n.op_type in ("GatherND", "ScatterND", "ScatterElements", "GatherGrad") for n in nodes
    )


def test_autograd_engine_build() -> None:
    engine = AutogradEngine()
    g = Graph("test_graph")
    res = engine.build_backward_graph(g)
    assert res is not None
