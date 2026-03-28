import pytest
import re
import logging
import struct
from onnx9000.core.ir import Graph, Constant, Node, SparseTensor
from onnx9000.core.dtypes import DType
from onnx9000.optimizer.sparse.modifier import (
    MagnitudePruningModifier,
    GradualPruningModifier,
    OBSPruningModifier,
    FisherPruningModifier,
    MovementPruningModifier,
    AccuracyAwarePruningModifier,
    AsymmetricSparseQuantizationModifier,
    SparseQLinearConvModifier,
    GlobalMagnitudePruningModifier,
    QuantizationModifier,
    NMPruningModifier,
    Modifier,
    parse_recipe,
    apply_recipe,
    ConstantPruningModifier,
)


def create_simple_graph():
    g = Graph("test")
    # Add a weight tensor
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    binary_data = struct.pack("<8f", *data)
    w1 = Constant("weight1", values=binary_data, shape=(2, 4), dtype=DType.FLOAT32)
    g.add_tensor(w1)
    g.initializers.append("weight1")
    return g


def test_modifier_absolute_edge_cases():
    g = create_simple_graph()
    # Tensor with None data
    w_none = Constant("w_none", values=None, shape=(4,), dtype=DType.FLOAT32)
    g.add_tensor(w_none)

    # Test all modifiers with None data to hit all "if tensor.data is None" lines
    modifiers = [
        ConstantPruningModifier(params=["re:w_none"]),
        MagnitudePruningModifier(params=["re:w_none"], final_sparsity=0.5),
        GradualPruningModifier(params=["re:w_none"], final_sparsity=0.5),
        OBSPruningModifier(params=["re:w_none"], sparsity=0.5),
        FisherPruningModifier(params=["re:w_none"], sparsity=0.5),
        MovementPruningModifier(params=["re:w_none"], sparsity=0.5),
        NMPruningModifier(params=["re:w_none"], n=2, m=4),
        QuantizationModifier(params=["re:w_none"]),
    ]
    for mod in modifiers:
        mod.apply(g)

    # Test with empty data (not None) to hit all "if not values" lines
    w_empty = Constant("w_empty", values=b"", shape=(0,), dtype=DType.FLOAT32)
    g.add_tensor(w_empty)
    for mod in modifiers:
        # Update params to match w_empty
        mod.params = ["re:w_empty"]
        mod.apply(g)


def test_modifier_base_final():
    m = Modifier(foo="bar")
    assert m.foo == "bar"
    with pytest.raises(NotImplementedError):
        m.apply(Graph("t"))


def test_magnitude_pruning_various_final():
    g = create_simple_graph()
    # Sparsity 1.0
    MagnitudePruningModifier(params=["re:weight1"], final_sparsity=1.0).apply(g)
    # Sparsity 0.0
    MagnitudePruningModifier(params=["re:weight1"], final_sparsity=0.0).apply(g)
    # L2
    MagnitudePruningModifier(params=["re:weight1"], final_sparsity=0.5, l2=True).apply(g)
    # Prevent zeroed
    MagnitudePruningModifier(
        params=["re:weight1"], final_sparsity=0.9, prevent_zeroed_channels=True
    ).apply(g)


def test_gradual_pruning_steps_final():
    g = create_simple_graph()
    mod = GradualPruningModifier(params=["re:weight1"], start_step=10, end_step=20)
    mod.current_step = 5
    mod.apply(g)
    mod.current_step = 15
    mod.apply(g)
    mod.current_step = 25
    mod.apply(g)


def test_obs_fisher_movement_final():
    g = create_simple_graph()
    OBSPruningModifier(params=["re:weight1"], sparsity=0.5, calibration_data=[[1.0] * 8]).apply(g)
    grads = {"weight1": [0.1] * 8}
    FisherPruningModifier(params=["re:weight1"], sparsity=0.5, gradients=grads).apply(g)
    MovementPruningModifier(params=["re:weight1"], sparsity=0.5, gradients=grads).apply(g)


def test_accuracy_aware_final():
    g = create_simple_graph()
    mod = AccuracyAwarePruningModifier(
        params=["re:weight1"], min_accuracy=0.95, target_sparsity=0.8
    )
    mod.apply(g, 0.90)
    mod.apply(g, 0.99)
    mod.current_sparsity = 0.9
    mod.apply(g, 0.99)


def test_global_magnitude_final():
    g = create_simple_graph()
    GlobalMagnitudePruningModifier(params=["re:weight1"], final_sparsity=0.5).apply(g)
    GlobalMagnitudePruningModifier(params=["re:weight1"], final_sparsity=0.5, l2=True).apply(g)
    GlobalMagnitudePruningModifier(
        params=["re:weight1"], final_sparsity=0.9, prevent_zeroed_channels=True
    ).apply(g)


def test_quantization_final():
    g = create_simple_graph()
    QuantizationModifier(params=["re:weight1"], scheme="asymmetric").apply(g)
    w = g.tensors["weight1"]
    w.data = struct.pack("<8f", *([0.0] * 8))
    QuantizationModifier(params=["re:weight1"]).apply(g)


def test_nm_pruning_final():
    g = create_simple_graph()
    NMPruningModifier(params=["re:weight1"], n=2, m=4).apply(g)
    w = g.tensors["weight1"]
    w.shape = (4, 2)
    with pytest.raises(ValueError):
        NMPruningModifier(params=["re:weight1"], m=4).apply(g)


def test_sparse_qlinear_conv_final():
    g = create_simple_graph()
    s = SparseTensor("s", dims=(2, 4))
    g.add_tensor(s)
    g.nodes.append(Node("Conv", ["in", "s"], ["out"]))
    SparseQLinearConvModifier().apply(g)


def test_recipe_final():
    g = create_simple_graph()
    apply_recipe(g, "- !MagnitudePruningModifier\n  params: ['re:weight1']")
    apply_recipe(g, [MagnitudePruningModifier(params=["re:weight1"])])


def test_other_stubs():
    from onnx9000.optimizer.sparse.modifier import manage_calibration_memory

    manage_calibration_memory(Graph("t"))
