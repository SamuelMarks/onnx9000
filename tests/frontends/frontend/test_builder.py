"""Module providing core logic and structural definitions."""

import pytest
from onnx9000.frontends.frontend.builder import (
    GraphBuilder,
    Tracing,
    get_active_builder,
)
from onnx9000.frontends.frontend.tensor import Node, Tensor
from onnx9000.core.dtypes import DType
from onnx9000.frontends.frontend.jit import jit


def test_graph_builder():
    """Provides semantic functionality and verification."""
    gb = GraphBuilder("test")
    assert gb.name == "test"
    n = Node("Relu", ["in"], ["out"])
    gb.add_node(n)
    assert len(gb.nodes) == 1


def test_tracing_context():
    """Provides semantic functionality and verification."""
    assert get_active_builder() is None
    gb = GraphBuilder("test")
    with Tracing(gb) as b:
        assert get_active_builder() is gb
        assert b is gb
    assert get_active_builder() is None


def test_jit_decorator():
    """Provides semantic functionality and verification."""

    @jit
    def my_func(x):
        """Provides my func functionality and verification."""
        return x + x

    t = Tensor((10,), DType.FLOAT32, "x")
    res = my_func(t)
    assert isinstance(res, GraphBuilder)
    assert res.name == "my_func"
    assert len(res.inputs) == 1
    assert len(res.nodes) == 1
    assert len(res.outputs) == 1


def test_jit_multi_out():
    """Provides semantic functionality and verification."""

    @jit
    def my_func(x):
        """Provides my func functionality and verification."""
        return x + x, x * x

    t = Tensor((10,), DType.FLOAT32, "x")
    res = my_func(t)
    assert len(res.outputs) == 2


def test_jit_list_out():
    """Provides semantic functionality and verification."""

    @jit
    def my_func(x):
        """Provides my func functionality and verification."""
        return [x + x, x * x]

    t = Tensor((10,), DType.FLOAT32, "x")
    res = my_func(t)
    assert len(res.outputs) == 2
