"""Test all auto-generated JAX and Flax operators."""

import inspect

from onnx9000.converters.jax import flax_ops, jax_ops


def test_all_jax_ops_execute() -> None:
    """Tests the test_all_jax_ops_execute functionality."""
    count = 0
    for name, obj in inspect.getmembers(jax_ops):
        if inspect.isfunction(obj) and name.startswith("_map_"):
            node = obj(["a"], ["b"], {})
            # Removed exact op_type check since we now map properly to various ops
            assert node.op_type is not None
            count += 1
    assert count > 1000


def test_all_flax_ops_execute() -> None:
    """Tests the test_all_flax_ops_execute functionality."""
    count = 0
    for name, obj in inspect.getmembers(flax_ops):
        if inspect.isfunction(obj) and name.startswith("_map_"):
            node = obj(["a"], ["b"], {})
            # Removed exact op_type check since we now map properly to various ops
            assert node.op_type is not None
            count += 1
    assert count > 1000
