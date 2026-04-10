"""Tests for jax pytorch."""

import pytest
from onnx9000.converters.jax.jax_ops import _map_jax_add_prim
from onnx9000.converters.pytorch_fx_parser import load_pytorch_fx
from onnx9000.core.registry import global_registry


def test_jax_mapped():
    """Docstring for D103."""
    op = global_registry.get_op("jax", "add")
    node = op(inputs=["x", "y"], outputs=["z"], params={})
    assert node.op_type == "Add"


def test_pytorch_mapped():
    """Docstring for D103."""
    op = global_registry.get_op("pytorch", "aten.convolution.default")
    node = op(inputs=["x", "w"], outputs=["z"], params={})
    assert node.op_type == "Conv"


def test_pytorch_parser():
    """Docstring for D103."""
    json_str = '{"nodes": [{"op": "call_function", "target": "aten.add.Tensor", "args": ["x", "y"], "out": "z"}]}'
    graph = load_pytorch_fx(json_str)
    assert graph.nodes[0].op_type == "Add"
