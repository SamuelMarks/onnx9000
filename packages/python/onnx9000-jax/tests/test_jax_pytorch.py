"""Tests for jax pytorch."""

from onnx9000.converters.pytorch_fx_parser import load_pytorch_fx
from onnx9000.core.registry import global_registry


def test_jax_mapped() -> None:
    """Verify that jax mapped behaves as expected."""
    op = global_registry.get_op("jax", "add")
    node = op(inputs=["x", "y"], outputs=["z"], params={})
    assert node.op_type == "Add"


def test_pytorch_mapped() -> None:
    """Verify that pytorch mapped behaves as expected."""
    op = global_registry.get_op("pytorch", "aten.convolution.default")
    node = op(inputs=["x", "w"], outputs=["z"], params={})
    assert node.op_type == "Conv"


def test_pytorch_parser() -> None:
    """Verify that pytorch parser behaves as expected."""
    json_str = '{"nodes": [{"op": "call_function", "target": "aten.add.Tensor", "args": ["x", "y"], "out": "z"}]}'
    graph = load_pytorch_fx(json_str)
    assert graph.nodes[0].op_type == "Add"
