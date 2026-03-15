"""Module providing core logic and structural definitions."""

from onnx9000.training.autograd.compiler import AutogradEngine
from onnx9000.core.ir import Graph


def test_autograd_engine_no_grad():
    """Provides semantic functionality and verification."""
    engine = AutogradEngine()
    assert not engine._no_grad
    with engine.no_grad():
        assert engine._no_grad
    assert not engine._no_grad


def test_autograd_engine_build():
    """Provides semantic functionality and verification."""
    engine = AutogradEngine()
    g = Graph("test_graph")
    res = engine.build_backward_graph(g)
    assert res is not None
