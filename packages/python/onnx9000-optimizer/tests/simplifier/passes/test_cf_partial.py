"""Module docstring."""

import numpy as np
from onnx9000.core.ir import Node
from onnx9000.optimizer.simplifier.passes.constant_folding import ConstantFoldingPass


def test_cf_partial_fold():
    """Docstring for D103."""
    cf = ConstantFoldingPass()

    # Add
    n = Node("Add", ["in1", "in2"], ["out"])
    known = {"in2": np.array([0.0])}
    n_new, changed = cf._partial_fold(n, known)
    assert changed
    assert n_new.op_type == "Identity"
    assert n_new.inputs == ["in1"]

    n2 = Node("Add", ["in1", "in2"], ["out"])
    known2 = {"in1": np.array([0.0])}
    n_new2, changed2 = cf._partial_fold(n2, known2)
    assert changed2
    assert n_new2.op_type == "Identity"
    assert n_new2.inputs == ["in2"]

    # Mul
    n3 = Node("Mul", ["in1", "in2"], ["out"])
    known3 = {"in2": np.array([1.0])}
    n_new3, changed3 = cf._partial_fold(n3, known3)
    assert changed3
    assert n_new3.op_type == "Identity"
    assert n_new3.inputs == ["in1"]

    n4 = Node("Mul", ["in1", "in2"], ["out"])
    known4 = {"in1": np.array([1.0])}
    n_new4, changed4 = cf._partial_fold(n4, known4)
    assert changed4
    assert n_new4.op_type == "Identity"
    assert n_new4.inputs == ["in2"]

    n5 = Node("Mul", ["in1", "in2"], ["out"])
    known5 = {"in2": np.array([0.0])}
    n_new5, changed5 = cf._partial_fold(n5, known5)
    assert changed5
    assert n_new5.op_type == "Constant"

    # Sub
    n6 = Node("Sub", ["in1", "in2"], ["out"])
    known6 = {"in2": np.array([0.0])}
    n_new6, changed6 = cf._partial_fold(n6, known6)
    assert changed6
    assert n_new6.op_type == "Identity"

    n6b = Node("Sub", ["in1", "in1"], ["out"])
    n_new6b, changed6b = cf._partial_fold(n6b, {})
    assert changed6b
    assert n_new6b.op_type == "Constant"

    # Where
    n8 = Node("Where", ["cond", "x", "y"], ["out"])
    n_new8, changed8 = cf._partial_fold(n8, {"cond": np.array([True])})
    assert changed8
    assert n_new8.inputs == ["x"]

    n9 = Node("Where", ["cond", "x", "y"], ["out"])
    n_new9, changed9 = cf._partial_fold(n9, {"cond": np.array([False])})
    assert changed9
    assert n_new9.inputs == ["y"]

    # Div
    n10 = Node("Div", ["in1", "in2"], ["out"])
    n_new10, changed10 = cf._partial_fold(n10, {"in2": np.array([1.0])})
    assert changed10
    assert n_new10.op_type == "Identity"

    n11 = Node("Div", ["in1", "in1"], ["out"])
    n_new11, changed11 = cf._partial_fold(n11, {})
    assert changed11
    assert n_new11.op_type == "Constant"

    # Pow
    n12 = Node("Pow", ["in1", "in2"], ["out"])
    n_new12, changed12 = cf._partial_fold(n12, {"in2": np.array([1.0])})
    assert changed12
    assert n_new12.op_type == "Identity"

    n13 = Node("Pow", ["in1", "in2"], ["out"])
    n_new13, changed13 = cf._partial_fold(n13, {"in2": np.array([0.0])})
    assert changed13
    assert n_new13.op_type == "Constant"

    # Other
    n7 = Node("Other", ["in1", "in2"], ["out"])
    n_new7, changed7 = cf._partial_fold(n7, {})
    assert not changed7
