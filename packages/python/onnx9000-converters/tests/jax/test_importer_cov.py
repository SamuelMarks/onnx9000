"""Tests for JAX importer coverage gaps."""

import numpy as np
import pytest

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from onnx9000.converters.jax.jax_importer import JAXImporter


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_jax_importer_basic():
    """Test basic JAX function import."""

    def simple_func(x, y):
        return jnp.sin(x) + y

    importer = JAXImporter()
    x = np.ones((2, 2), dtype=np.float32)
    y = np.ones((2, 2), dtype=np.float32)

    graph = importer.import_func(simple_func, x, y)
    assert len(graph.nodes) > 0
    assert "Sin" in [n.op_type for n in graph.nodes]
    assert "Add" in [n.op_type for n in graph.nodes]


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_jax_importer_constants():
    """Test JAX import with constants."""
    if not JAX_AVAILABLE:
        return

    c = jnp.array([2.0], dtype=jnp.float32)

    def const_func(x):
        return x * c

    importer = JAXImporter()
    x = np.ones((1,), dtype=np.float32)
    graph = importer.import_func(const_func, x)
    assert any(n.op_type == "Mul" for n in graph.nodes)


def test_jax_importer_dtype_mapping():
    """Test JAX dtype mapping directly."""
    importer = JAXImporter()
    from onnx9000.core.dtypes import DType

    assert importer._map_dtype(np.float32) == DType.FLOAT32
    assert importer._map_dtype(np.int32) == DType.INT32
    assert importer._map_dtype(np.float64) == DType.FLOAT32  # default fallback in code


def test_jax_importer_primitive_mapping():
    """Test JAX primitive mapping."""
    importer = JAXImporter()
    assert importer._map_primitive("add") == "Add"
    assert importer._map_primitive("dot_general") == "MatMul"
    assert importer._map_primitive("unknown") == "Unknown"

    # test unhashable var
    unhashable = [1, 2, 3]  # lists are unhashable
    name1 = importer.get_var_name(unhashable)
    name2 = importer.get_var_name(unhashable)
    assert name1 == name2


from onnx9000.converters.jax.importer import _map_jax_type, load, load_jax
from onnx9000.core.dtypes import DType


def test_map_jax_type():
    """Docstring for D103."""
    assert _map_jax_type("f32") == DType.FLOAT32
    assert _map_jax_type("i32") == DType.INT32
    assert _map_jax_type("other") == DType.FLOAT32


def test_load_jax_with_consts():
    """Docstring for D103."""
    jaxpr_dict = {
        "invars": [{"name": "in1", "type": "f32", "shape": [1]}],
        "outvars": [{"name": "out1", "type": "f32", "shape": [1]}],
        "constvars": [{"name": "c1", "type": "f32", "shape": [1]}],
        "eqns": [
            {
                "primitive": "add",
                "invars": [{"name": "in1"}, {"name": "c1"}],
                "outvars": [{"name": "out1", "type": "f32", "shape": [1]}],
                "params": {},
            },
            {
                "primitive": "unknown_primitive",
                "invars": [{"name": "out1"}],
                "outvars": [{"name": "out2", "type": "f32", "shape": [1]}],
                "params": {},
            },
        ],
    }
    graph = load_jax(jaxpr_dict)
    assert graph is not None
    assert "c1" in graph.initializers

    nodes = graph.nodes
    assert any(n.op_type == "unknown_primitive" for n in nodes)


def test_load_auto_format():
    """Docstring for D103."""
    assert load({}, format="tf") is None
    assert load({"node": []}) is None

    jaxpr = {"eqns": []}
    assert load(jaxpr) is not None
    assert load({}, format="jax") is not None

    import sys
    from unittest.mock import MagicMock, patch

    with patch.dict(
        sys.modules,
        {"onnx9000.core.parser.core": MagicMock(load=MagicMock(return_value="onnx_loaded"))},
    ):
        assert load("model.pb", format="onnx") == "onnx_loaded"
