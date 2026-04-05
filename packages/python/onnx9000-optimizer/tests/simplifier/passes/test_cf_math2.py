"""Tests for cf math2."""

import numpy as np
from onnx9000.optimizer.simplifier.passes.constant_folding import ConstantFoldingPass


def test_cf_math_ops2():
    """Docstring for D103."""
    cf = ConstantFoldingPass()
    a = np.array([2.0], dtype=np.float32)
    b = np.array([3.0], dtype=np.float32)

    # Cast
    assert cf._evaluate_node("Cast", [a], {"to": 6}) is not None
    assert cf._evaluate_node("Cast", [a], {"to": 999}) is not None

    # Reshape, Transpose, Squeeze, Unsqueeze
    assert cf._evaluate_node("Reshape", [a, np.array([1])], {}) is not None
    assert cf._evaluate_node("Transpose", [np.ones((2, 2))], {"perm": [1, 0]}) is not None
    assert cf._evaluate_node("Squeeze", [np.ones((1, 2))], {"axes": [0]}) is not None
    assert cf._evaluate_node("Squeeze", [np.ones((1, 2)), np.array([0])], {}) is not None
    assert cf._evaluate_node("Unsqueeze", [np.ones((2,))], {"axes": [0]}) is not None
    assert cf._evaluate_node("Unsqueeze", [np.ones((2,)), np.array([0])], {}) is not None

    # Flatten, Concat, Slice
    assert cf._evaluate_node("Flatten", [np.ones((2, 2, 2))], {"axis": 1}) is not None
    assert cf._evaluate_node("Concat", [np.ones((2, 2)), np.ones((2, 2))], {"axis": 0}) is not None

    # slice
    data = np.arange(10)
    starts = np.array([0])
    ends = np.array([5])
    assert cf._evaluate_node("Slice", [data, starts, ends], {}) is not None
    assert (
        cf._evaluate_node("Slice", [data, starts, ends, np.array([0]), np.array([2])], {})
        is not None
    )

    # BN
    X = np.ones((1, 2, 2, 2))
    scale = np.ones((2,))
    b = np.ones((2,))
    mean = np.ones((2,))
    var = np.ones((2,))
    assert (
        cf._evaluate_node("BatchNormalization", [X, scale, b, mean, var], {"epsilon": 1e-05})
        is not None
    )

    X1d = np.ones((2,))
    assert cf._evaluate_node("BatchNormalization", [X1d, scale, b, mean, var], {}) is not None

    # Split
    assert cf._evaluate_node("Split", [np.ones((4,)), np.array([2, 2])], {"axis": 0}) is not None

    class NodeMock:
        """Node mock."""

        outputs = ["a", "b"]

    assert cf._evaluate_node("Split", [np.ones((4,))], {"axis": 0}, node=NodeMock()) is not None

    # Expand, Tile
    assert cf._evaluate_node("Expand", [np.ones((2,)), np.array([2, 2])], {}) is not None
    assert cf._evaluate_node("Tile", [np.ones((2,)), np.array([2])], {}) is not None

    # Pad
    assert (
        cf._evaluate_node("Pad", [np.ones((2,)), np.array([1, 1])], {"mode": "constant"})
        is not None
    )
    assert (
        cf._evaluate_node(
            "Pad", [np.ones((2,)), np.array([1, 1]), np.array([5])], {"mode": "constant"}
        )
        is not None
    )
    assert (
        cf._evaluate_node(
            "Pad", [np.ones((2,)), np.array([1, 1]), np.array([5, 5])], {"mode": "constant"}
        )
        is not None
    )
    assert (
        cf._evaluate_node("Pad", [np.ones((2,)), np.array([1, 1])], {"mode": "reflect"}) is not None
    )
    assert cf._evaluate_node("Pad", [np.ones((2,)), np.array([1, 1])], {"mode": "edge"}) is not None
    assert cf._evaluate_node("Pad", [np.ones((2,)), np.array([1, 1])], {"mode": "other"}) is None

    # ConstantOfShape
    assert (
        cf._evaluate_node(
            "ConstantOfShape", [np.array([2])], {"value": np.array([1.0], dtype=np.float32)}
        )
        is not None
    )
    from onnx9000.core.dtypes import DType
    from onnx9000.core.ir import Tensor

    t = Tensor("t", (1,), DType.FLOAT32, data=np.array([1.0], dtype=np.float32).tobytes())
    assert cf._evaluate_node("ConstantOfShape", [np.array([2])], {"value": t}) is not None
    assert (
        cf._evaluate_node("ConstantOfShape", [], {"value": np.array([1.0], dtype=np.float32)})
        is not None
    )
    assert (
        cf._evaluate_node(
            "ConstantOfShape",
            [np.array([2])],
            {"value": np.array([1.0], dtype=np.float32).tobytes()},
        )
        is not None
    )
    assert cf._evaluate_node("ConstantOfShape", [np.array([2])], {"value": [1.0]}) is not None

    # Where, CumSum, Trilu, Gemm
    assert (
        cf._evaluate_node("Where", [np.array([True, False]), np.array([1]), np.array([2])], {})
        is not None
    )

    # Cumsum
    assert cf._evaluate_node("CumSum", [np.array([1, 2, 3]), np.array([0])], {}) is not None
    assert (
        cf._evaluate_node(
            "CumSum", [np.array([1, 2, 3]), np.array([0])], {"exclusive": 1, "reverse": 1}
        )
        is not None
    )

    # Trilu
    assert cf._evaluate_node("Trilu", [np.ones((2, 2))], {}) is not None
    assert cf._evaluate_node("Trilu", [np.ones((2, 2)), np.array([1])], {"upper": 0}) is not None

    # Gemm/Matmul
    assert cf._evaluate_node("MatMul", [np.ones((2, 2)), np.ones((2, 2))], {}) is not None
    assert cf._evaluate_node("Gemm", [np.ones((2, 2)), np.ones((2, 2))], {}) is not None
    assert (
        cf._evaluate_node(
            "Gemm",
            [np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2))],
            {"transA": 1, "transB": 1, "alpha": 2.0, "beta": 3.0},
        )
        is not None
    )
