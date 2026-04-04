"""Module docstring."""

import numpy as np
from onnx9000.optimizer.simplifier.passes.constant_folding import ConstantFoldingPass


def test_cf_math_ops():
    """Docstring for D103."""
    cf = ConstantFoldingPass()
    a = np.array([2.0], dtype=np.float32)
    b = np.array([3.0], dtype=np.float32)

    assert cf._evaluate_node("Add", [a, b], {}) == 5.0
    assert cf._evaluate_node("Sub", [a, b], {}) == -1.0
    assert cf._evaluate_node("Mul", [a, b], {}) == 6.0
    assert cf._evaluate_node("Div", [a, b], {}) == 2.0 / 3.0
    assert cf._evaluate_node("Pow", [a, b], {}) == 8.0

    a_neg = np.array([-2.0], dtype=np.float32)
    assert cf._evaluate_node("Abs", [a_neg], {}) == 2.0
    assert cf._evaluate_node("Exp", [a], {}) is not None
    assert cf._evaluate_node("Log", [a], {}) is not None
    assert cf._evaluate_node("Sqrt", [a], {}) is not None
    assert cf._evaluate_node("Sin", [a], {}) is not None
    assert cf._evaluate_node("Cos", [a], {}) is not None
    assert cf._evaluate_node("Tan", [a], {}) is not None
    assert cf._evaluate_node("Asin", [np.array([0.5])], {}) is not None
    assert cf._evaluate_node("Acos", [np.array([0.5])], {}) is not None
    assert cf._evaluate_node("Atan", [a], {}) is not None
    assert cf._evaluate_node("Sinh", [a], {}) is not None
    assert cf._evaluate_node("Cosh", [a], {}) is not None
    assert cf._evaluate_node("Tanh", [a], {}) is not None
    assert cf._evaluate_node("Neg", [a], {}) == -2.0
    assert cf._evaluate_node("Sign", [a_neg], {}) == -1.0
    assert cf._evaluate_node("Ceil", [np.array([1.5])], {}) == 2.0
    assert cf._evaluate_node("Floor", [np.array([1.5])], {}) == 1.0
    assert cf._evaluate_node("Round", [np.array([1.5])], {}) == 2.0
    assert cf._evaluate_node("Mod", [np.array([5.0]), np.array([2.0])], {}) == 1.0

    assert cf._evaluate_node("Max", [a, b], {}) == 3.0
    assert cf._evaluate_node("Max", [a], {}) == 2.0
    assert cf._evaluate_node("Min", [a, b], {}) == 2.0
    assert cf._evaluate_node("Min", [a], {}) == 2.0

    assert cf._evaluate_node("Clip", [a, np.array([0.0]), np.array([1.0])], {}) == 1.0

    bool_t = np.array([True])
    bool_f = np.array([False])
    assert not cf._evaluate_node("And", [bool_t, bool_f], {})
    assert cf._evaluate_node("Or", [bool_t, bool_f], {})
    assert not cf._evaluate_node("Not", [bool_t], {})
    assert not cf._evaluate_node("Xor", [bool_t, bool_t], {})

    assert cf._evaluate_node("Equal", [a, a], {})
    assert cf._evaluate_node("Greater", [b, a], {})
    assert cf._evaluate_node("Less", [a, b], {})
    assert cf._evaluate_node("GreaterOrEqual", [a, a], {})
    assert cf._evaluate_node("LessOrEqual", [a, a], {})

    int_a = np.array([4])
    int_b = np.array([1])
    assert cf._evaluate_node("BitShift", [int_a, int_b], {"direction": "RIGHT"}) == 2
    assert cf._evaluate_node("BitShift", [int_a, int_b], {"direction": "LEFT"}) == 8
    assert cf._evaluate_node("BitShift", [int_a, int_b], {"direction": "OTHER"}) is None

    assert cf._evaluate_node("BitwiseAnd", [np.array([3]), np.array([1])], {}) == 1
    assert cf._evaluate_node("BitwiseOr", [np.array([1]), np.array([2])], {}) == 3
    assert cf._evaluate_node("BitwiseNot", [np.array([0])], {}) == -1
    assert cf._evaluate_node("BitwiseXor", [np.array([3]), np.array([1])], {}) == 2

    inf_val = np.array([float("inf")])
    assert cf._evaluate_node("IsInf", [inf_val], {})
    assert cf._evaluate_node("IsInf", [inf_val], {"detect_positive": 1, "detect_negative": 0})
    assert cf._evaluate_node(
        "IsInf", [np.array([float("-inf")])], {"detect_positive": 0, "detect_negative": 1}
    )
    assert not cf._evaluate_node("IsInf", [a], {"detect_positive": 0, "detect_negative": 0})

    assert cf._evaluate_node("IsNaN", [np.array([float("nan")])], {})

    assert cf._evaluate_node("Erf", [a], {}) is not None
    assert cf._evaluate_node("Relu", [a_neg], {}) == 0.0
    assert cf._evaluate_node("Sigmoid", [np.array([0.0])], {}) == 0.5
