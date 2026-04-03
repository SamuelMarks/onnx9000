import numpy as np
from onnx9000.optimizer.simplifier.passes.constant_folding import ConstantFoldingPass


def test_cf_reductions():
    cf = ConstantFoldingPass()
    a = np.ones((2, 2))
    b = np.array([0])

    assert cf._evaluate_node("ReduceSum", [a, b], {}) is not None
    assert cf._evaluate_node("ReduceMean", [a, b], {}) is not None
    assert cf._evaluate_node("ReduceMax", [a, b], {}) is not None
    assert cf._evaluate_node("ReduceMin", [a, b], {}) is not None
    assert cf._evaluate_node("ReduceProd", [a, b], {}) is not None
    assert cf._evaluate_node("ReduceL1", [a, b], {}) is not None
    assert cf._evaluate_node("ReduceL2", [a, b], {}) is not None
    assert cf._evaluate_node("ReduceLogSum", [a, b], {}) is not None
    assert cf._evaluate_node("ReduceLogSumExp", [a, b], {}) is not None

    # Hit keepdims=0
    assert cf._evaluate_node("ReduceSum", [a, b], {"keepdims": 0}) is not None
    assert cf._evaluate_node("ReduceMean", [a, b], {"keepdims": 0}) is not None
    assert cf._evaluate_node("ReduceMax", [a, b], {"keepdims": 0}) is not None
    assert cf._evaluate_node("ReduceMin", [a, b], {"keepdims": 0}) is not None
    assert cf._evaluate_node("ReduceProd", [a, b], {"keepdims": 0}) is not None
    assert cf._evaluate_node("ReduceL1", [a, b], {"keepdims": 0}) is not None
    assert cf._evaluate_node("ReduceL2", [a, b], {"keepdims": 0}) is not None
    assert cf._evaluate_node("ReduceLogSum", [a, b], {"keepdims": 0}) is not None
    assert cf._evaluate_node("ReduceLogSumExp", [a, b], {"keepdims": 0}) is not None

    # Hit axes in attrs
    assert cf._evaluate_node("ReduceSum", [a], {"axes": [0]}) is not None
    assert cf._evaluate_node("ReduceMean", [a], {"axes": [0]}) is not None
    assert cf._evaluate_node("ReduceMax", [a], {"axes": [0]}) is not None
    assert cf._evaluate_node("ReduceMin", [a], {"axes": [0]}) is not None
    assert cf._evaluate_node("ReduceProd", [a], {"axes": [0]}) is not None
    assert cf._evaluate_node("ReduceL1", [a], {"axes": [0]}) is not None
    assert cf._evaluate_node("ReduceL2", [a], {"axes": [0]}) is not None
    assert cf._evaluate_node("ReduceLogSum", [a], {"axes": [0]}) is not None
    assert cf._evaluate_node("ReduceLogSumExp", [a], {"axes": [0]}) is not None

    # Hit missing CastLike (602-603)
    b_cast = np.array([2.0], dtype=np.float32)
    assert cf._evaluate_node("CastLike", [a, b_cast], {}) is not None
