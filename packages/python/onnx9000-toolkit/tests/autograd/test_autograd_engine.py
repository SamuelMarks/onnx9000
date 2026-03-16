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


def test_autograd_engine_build() -> None:
    """Tests the test_autograd_engine_build functionality."""
    engine = AutogradEngine()
    g = Graph("test_graph")
    res = engine.build_backward_graph(g)
    assert res is not None
