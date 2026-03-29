"""Tests for sparse modifiers."""

import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Constant, Graph, Node
from onnx9000.core.sparse import pack_data, unpack_data
from onnx9000.optimizer.sparse.modifier import (
    GlobalMagnitudePruningModifier,
    MagnitudePruningModifier,
    NMPruningModifier,
    apply_recipe,
    parse_recipe,
)


def create_simple_graph():
    """Tests the simple graph creation functionality."""
    g = Graph("test")
    # Weights: 1, 2, 3, 4, 5, 6, 7, 8
    data = pack_data([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], DType.FLOAT32)
    w = Constant("weight1", values=data, shape=(2, 4), dtype=DType.FLOAT32)
    g.add_tensor(w)

    node = Node("MatMul", inputs=["input", "weight1"], outputs=["output"], name="matmul1")
    g.add_node(node)
    return g


def test_magnitude_pruning():
    """Tests the magnitude pruning functionality."""
    g = create_simple_graph()
    # Prune 50% -> threshold should be 4.0 (keeps 5, 6, 7, 8)
    mod = MagnitudePruningModifier(params=["re:weight1"], final_sparsity=0.5)
    mod.apply(g)

    w = g.tensors["weight1"]
    vals = unpack_data(w.data, w.dtype)
    assert vals == [0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 7.0, 8.0]


def test_global_magnitude_pruning():
    """Tests the global magnitude pruning functionality."""
    g = Graph("test_global")
    data1 = pack_data([1.0, 10.0], DType.FLOAT32)
    data2 = pack_data([2.0, 20.0], DType.FLOAT32)
    w1 = Constant("w1", values=data1, shape=(2,), dtype=DType.FLOAT32)
    w2 = Constant("w2", values=data2, shape=(2,), dtype=DType.FLOAT32)
    g.add_tensor(w1)
    g.add_tensor(w2)

    # Total values: 1, 10, 2, 20. Prune 50% -> keeps 10, 20.
    mod = GlobalMagnitudePruningModifier(params=["re:w.*"], final_sparsity=0.5)
    mod.apply(g)

    vals1 = unpack_data(g.tensors["w1"].data, DType.FLOAT32)
    vals2 = unpack_data(g.tensors["w2"].data, DType.FLOAT32)
    assert vals1 == [0.0, 10.0]
    assert vals2 == [0.0, 20.0]


def test_nm_pruning():
    """Tests the NM pruning functionality."""
    g = create_simple_graph()
    # 2:4 pruning. Blocks of 4.
    # Block 1: 1, 2, 3, 4 -> keep 3, 4
    # Block 2: 5, 6, 7, 8 -> keep 7, 8
    mod = NMPruningModifier(params=["re:weight1"], n=2, m=4)
    mod.apply(g)

    w = g.tensors["weight1"]
    vals = unpack_data(w.data, w.dtype)
    assert vals == [0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 7.0, 8.0]


def test_parse_recipe():
    """Tests the sparse recipe parsing functionality."""
    recipe = """
    version: 1.1.0
    modifiers:
        - !MagnitudePruningModifier
            init_sparsity: 0.0
            final_sparsity: 0.8
            params: ['re:.*weight']
        - !NMPruningModifier
            n: 2
            m: 4
            params: ['conv1.weight']
    """
    modifiers = parse_recipe(recipe)
    assert len(modifiers) == 2
    assert isinstance(modifiers[0], MagnitudePruningModifier)
    assert modifiers[0].final_sparsity == 0.8
    assert modifiers[0].params == ["re:.*weight"]
    assert isinstance(modifiers[1], NMPruningModifier)
    assert modifiers[1].n == 2
    assert modifiers[1].m == 4


def test_apply_recipe():
    """Tests the sparse recipe application functionality."""
    g = create_simple_graph()
    recipe = """
    modifiers:
        - !MagnitudePruningModifier
            final_sparsity: 0.5
            params: ['re:weight1']
    """
    apply_recipe(g, recipe)
    w = g.tensors["weight1"]
    vals = unpack_data(w.data, w.dtype)
    assert vals == [0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 7.0, 8.0]
    assert g.metadata_props["onnx9000_sparse_recipe"] == recipe


def test_modifier_extra():
    import numpy as np
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Constant, Graph
    from onnx9000.optimizer.sparse.modifier import (
        ConstantPruningModifier,
        GlobalMagnitudePruningModifier,
        MagnitudePruningModifier,
    )

    g = Graph("g")
    g.tensors["t"] = Constant(
        "t",
        values=np.array([1.0, -0.5, 0.1, 0.05], dtype=np.float32).tobytes(),
        dtype=DType.FLOAT32,
        shape=(4,),
    )
    g.initializers.append("t")

    mod = ConstantPruningModifier(params=["t"], threshold=0.2)
    mod.apply(g)

    # Check unpacked data
    import struct

    vals = struct.unpack("<4f", g.tensors["t"].data)
    assert abs(vals[0] - 1.0) < 1e-5
    assert abs(vals[1] + 0.5) < 1e-5
    assert abs(vals[2]) < 1e-5
    assert abs(vals[3]) < 1e-5
