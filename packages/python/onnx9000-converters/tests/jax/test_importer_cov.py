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

    def const_func(x):
        return x * 2.0

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
