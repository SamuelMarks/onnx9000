"""Test real JAX primitive mappers."""

import inspect
from typing import Any
from onnx9000.converters.jax import jax_ops


def test_all_jax_ops_execute() -> None:
    """Tests the test_all_jax_ops_execute functionality."""
    count = 0
    for name, obj in inspect.getmembers(jax_ops):
        if inspect.isfunction(obj) and name.startswith("_map_"):
            node = obj(["a"], ["b"], {})
            assert node.op_type is not None
            count += 1
    assert count > 0  # Should be around 20 real ops now
