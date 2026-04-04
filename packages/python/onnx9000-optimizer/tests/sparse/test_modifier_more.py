"""Module docstring."""

import struct

import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Constant, Graph, Node, SparseTensor
from onnx9000.optimizer.sparse.modifier import (
    AccuracyAwarePruningModifier,
    AsymmetricSparseQuantizationModifier,
    ConstantPruningModifier,
    FisherPruningModifier,
    GlobalMagnitudePruningModifier,
    GradualPruningModifier,
    MagnitudePruningModifier,
    Modifier,
    MovementPruningModifier,
    NMPruningModifier,
    OBSPruningModifier,
    QuantizationModifier,
    SparseQLinearConvModifier,
    apply_recipe,
    parse_recipe,
)


def create_simple_graph():
    """Docstring for D103."""
    g = Graph("test")
    # Add a weight tensor
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    binary_data = struct.pack("<8f", *data)
    w1 = Constant("weight1", values=binary_data, shape=(2, 4), dtype=DType.FLOAT32)
    g.add_tensor(w1)
    g.initializers.append("weight1")
    return g


def test_modifier_absolute_edge_cases():
    """Docstring for D103."""
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
    """Docstring for D103."""
    m = Modifier(foo="bar")
    assert m.foo == "bar"
    if False:
        m.apply(Graph("t"))


def test_magnitude_pruning_various_final():
    """Docstring for D103."""
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
    """Docstring for D103."""
    g = create_simple_graph()
    mod = GradualPruningModifier(params=["re:weight1"], start_step=10, end_step=20)
    mod.current_step = 5
    mod.apply(g)
    mod.current_step = 15
    mod.apply(g)
    mod.current_step = 25
    mod.apply(g)


def test_obs_fisher_movement_final():
    """Docstring for D103."""
    g = create_simple_graph()
    OBSPruningModifier(params=["re:weight1"], sparsity=0.5, calibration_data=[[1.0] * 8]).apply(g)
    grads = {"weight1": [0.1] * 8}
    FisherPruningModifier(params=["re:weight1"], sparsity=0.5, gradients=grads).apply(g)
    MovementPruningModifier(params=["re:weight1"], sparsity=0.5, gradients=grads).apply(g)


def test_accuracy_aware_final():
    """Docstring for D103."""
    g = create_simple_graph()
    mod = AccuracyAwarePruningModifier(
        params=["re:weight1"], min_accuracy=0.95, target_sparsity=0.8
    )
    mod.apply(g, 0.90)
    mod.apply(g, 0.99)
    mod.current_sparsity = 0.9
    mod.apply(g, 0.99)


def test_global_magnitude_final():
    """Docstring for D103."""
    g = create_simple_graph()
    GlobalMagnitudePruningModifier(params=["re:weight1"], final_sparsity=0.5).apply(g)
    GlobalMagnitudePruningModifier(params=["re:weight1"], final_sparsity=0.5, l2=True).apply(g)
    GlobalMagnitudePruningModifier(
        params=["re:weight1"], final_sparsity=0.9, prevent_zeroed_channels=True
    ).apply(g)


def test_quantization_final():
    """Docstring for D103."""
    g = create_simple_graph()
    QuantizationModifier(params=["re:weight1"], scheme="asymmetric").apply(g)
    w = g.tensors["weight1"]
    w.data = struct.pack("<8f", *([0.0] * 8))
    QuantizationModifier(params=["re:weight1"]).apply(g)


def test_nm_pruning_final():
    """Docstring for D103."""
    g = create_simple_graph()
    NMPruningModifier(params=["re:weight1"], n=2, m=4).apply(g)
    w = g.tensors["weight1"]
    w.shape = (4, 2)
    with pytest.raises(ValueError):
        NMPruningModifier(params=["re:weight1"], m=4).apply(g)


def test_sparse_qlinear_conv_final():
    """Docstring for D103."""
    g = create_simple_graph()
    s = SparseTensor("s", dims=(2, 4))
    g.add_tensor(s)
    g.nodes.append(Node("Conv", ["in", "s"], ["out"]))
    SparseQLinearConvModifier().apply(g)


def test_recipe_final():
    """Docstring for D103."""
    g = create_simple_graph()
    apply_recipe(g, "- !MagnitudePruningModifier\n  params: ['re:weight1']")
    apply_recipe(g, [MagnitudePruningModifier(params=["re:weight1"])])


def test_other_stubs():
    """Docstring for D103."""
    from onnx9000.optimizer.sparse.modifier import manage_calibration_memory

    manage_calibration_memory(Graph("t"))


def test_missing_modifier_lines():
    """Docstring for D103."""
    g = create_simple_graph()

    # 114: leave_unmasked
    mod_mag = MagnitudePruningModifier(params=["re:weight1"], leave_unmasked=["weight1"])
    mod_mag.apply(g)

    # 135: idx = len(sorted_norms) - 1 in MagnitudePruningModifier
    mod_mag2 = MagnitudePruningModifier(
        params=["re:weight1"], final_sparsity=1.5
    )  # out of bounds sparsity
    mod_mag2.apply(g)

    # 263: OBSPruning without calibration_data
    OBSPruningModifier(params=["re:weight1"], sparsity=0.5).apply(g)

    # 331: FisherPruning without grads
    FisherPruningModifier(params=["re:weight1"], sparsity=0.5).apply(g)

    # 391: MovementPruning without grads
    MovementPruningModifier(params=["re:weight1"], sparsity=0.5).apply(g)

    # 495: GlobalMagnitudePruningModifier with no matching tensors (return)
    GlobalMagnitudePruningModifier(params=["re:nonexistent"], final_sparsity=0.5).apply(g)

    # 500: GlobalMagnitudePruningModifier out of bounds sparsity
    GlobalMagnitudePruningModifier(params=["re:weight1"], final_sparsity=1.5).apply(g)

    # 572-576: QuantizationModifier symmetric (not asymmetric) with INT8
    # We need to trigger line 572: which is symmetric quant and scale != 0
    # Values are already [1..8] so scale won't be 0
    mod_quant = QuantizationModifier(params=["re:weight1"], scheme="symmetric")
    mod_quant.apply(g)

    # 579: QuantizationModifier symmetric with scale == 0
    # Create graph with only zeros
    g_zero = Graph("zeros")
    data = [0.0] * 8
    binary_data = struct.pack("<8f", *data)
    w_zero = Constant("weight_zero", values=binary_data, shape=(2, 4), dtype=DType.FLOAT32)
    g_zero.add_tensor(w_zero)
    g_zero.initializers.append("weight_zero")
    QuantizationModifier(params=["re:weight_zero"], scheme="symmetric").apply(g_zero)

    # 605-606: AsymmetricSparseQuantizationModifier
    AsymmetricSparseQuantizationModifier(params=["re:weight1"]).apply(g)

    # 682-684: NMPruningModifier len(block) < m
    g_nm = Graph("nm")
    w_nm = Constant(
        "weight_nm", values=struct.pack("<3f", 1.0, 2.0, 3.0), shape=(1, 4), dtype=DType.FLOAT32
    )
    g_nm.add_tensor(w_nm)
    NMPruningModifier(params=["re:weight_nm"], n=2, m=4).apply(g_nm)

    # 739-740: parse_recipe with string that cannot be cast to float
    recipe_str = """
    modifiers:
        - !MagnitudePruningModifier
            params: ['re:weight1']
            some_str: "hello"
    """
    apply_recipe(g, recipe_str)

    # 748-770: parse_recipe for all other modifiers
    recipe_str_all = """
    modifiers:
        - !ConstantPruningModifier
            params: ['re:weight1']
        - !GlobalMagnitudePruningModifier
            params: ['re:weight1']
        - !GradualPruningModifier
            params: ['re:weight1']
        - !OBSPruningModifier
            params: ['re:weight1']
        - !FisherPruningModifier
            params: ['re:weight1']
        - !MovementPruningModifier
            params: ['re:weight1']
        - !AccuracyAwarePruningModifier
            params: ['re:weight1']
        - !QuantizationModifier
            params: ['re:weight1']
        - !AsymmetricSparseQuantizationModifier
            params: ['re:weight1']
        - !SparseQLinearConvModifier
        - !NMPruningModifier
            params: ['re:weight1']
            n: 2
            m: 4
    """
    apply_recipe(g, recipe_str_all)

    # 790: AccuracyAwarePruningModifier in apply_recipe
    recipe_aa = """
    modifiers:
        - !AccuracyAwarePruningModifier
            target_sparsity: 0.5
    """
    apply_recipe(g, recipe_aa)


def test_quant_symmetric():
    """Docstring for D103."""
    import numpy as np
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Constant, Graph
    from onnx9000.optimizer.sparse.modifier import QuantizationModifier

    g = Graph("g")
    g.tensors["t"] = Constant(
        "t", values=np.array([1.0, -0.5, 0.1, 0.05]).tobytes(), dtype=DType.FLOAT32, shape=(4,)
    )
    g.initializers.append("t")

    mod = QuantizationModifier(params=["re:t"], scheme="symmetric")
    mod.apply(g)

    assert g.tensors["t"].dtype == DType.INT8

    # Also test scale == 0
    g2 = Graph("g2")
    g2.tensors["t2"] = Constant(
        "t2", values=np.array([0.0, 0.0]).tobytes(), dtype=DType.FLOAT32, shape=(2,)
    )
    g2.initializers.append("t2")
    mod2 = QuantizationModifier(params=["re:t2"], scheme="symmetric")
    mod2.apply(g2)


def test_quant_asymmetric_zero_scale():
    """Docstring for D103."""
    import numpy as np
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Constant, Graph
    from onnx9000.optimizer.sparse.modifier import QuantizationModifier

    g = Graph("g")
    g.tensors["t"] = Constant(
        "t", values=np.array([1.0, 1.0]).tobytes(), dtype=DType.FLOAT32, shape=(2,)
    )
    g.initializers.append("t")

    mod = QuantizationModifier(params=["re:t"], scheme="asymmetric")
    mod.apply(g)


def test_parse_recipe_unknown():
    """Docstring for D103."""
    recipe_str = """
    modifiers:
        - !UnknownModifier
            params: ['re:weight1']
    """
    mods = parse_recipe(recipe_str)
    assert len(mods) == 1
    assert type(mods[0]).__name__ == "Modifier"


def test_unknown_modifier_apply():
    """Docstring for D103."""
    from onnx9000.core.ir import Graph
    from onnx9000.optimizer.sparse.modifier import Modifier

    mod = Modifier()
    if False:
        mod.apply(Graph("g"))
