import numpy as np
from onnx9000.optimizer.simplifier.passes.constant_folding import ConstantFoldingPass


def test_cf_gather_scatter_seq():
    cf = ConstantFoldingPass()

    # Gather
    assert (
        cf._evaluate_node("Gather", [np.array([1, 2, 3]), np.array([0])], {"axis": 0}) is not None
    )

    # GatherND
    data = np.array([[1, 2], [3, 4]])
    indices = np.array([[0, 0], [1, 1]])
    assert cf._evaluate_node("GatherND", [data, indices], {"batch_dims": 0}) is not None

    class AttrMock:
        value = 0

    assert cf._evaluate_node("GatherND", [data, indices], {"batch_dims": AttrMock()}) is not None

    # ScatterND
    updates = np.array([9, 9])
    assert cf._evaluate_node("ScatterND", [data, indices, updates], {}) is not None
    assert (
        cf._evaluate_node("ScatterND", [data, indices, updates], {"reduction": "mul"}) is not None
    )
    assert (
        cf._evaluate_node("ScatterND", [data, indices, updates], {"reduction": "max"}) is not None
    )
    assert (
        cf._evaluate_node("ScatterND", [data, indices, updates], {"reduction": "min"}) is not None
    )
    assert (
        cf._evaluate_node("ScatterND", [data, indices, updates], {"reduction": AttrMock()})
        is not None
    )

    # SequenceConstruct
    assert cf._evaluate_node("SequenceConstruct", [np.array([1])], {}) is not None

    # SequenceAt
    seq = [np.array([1]), np.array([2])]
    assert cf._evaluate_node("SequenceAt", [seq, np.array([0])], {}) is not None
    assert cf._evaluate_node("SequenceAt", [seq, np.array([99])], {}) is None

    # SplitToSequence
    assert (
        cf._evaluate_node("SplitToSequence", [np.ones((10,)), np.array([2])], {"axis": 0})
        is not None
    )
    assert (
        cf._evaluate_node("SplitToSequence", [np.ones((10,)), np.array([2])], {"axis": AttrMock()})
        is not None
    )
    assert (
        cf._evaluate_node("SplitToSequence", [np.ones((100000,)), np.array([2])], {"axis": 0})
        is None
    )

    # GatherElements
    assert (
        cf._evaluate_node("GatherElements", [data, np.array([[0, 0], [0, 0]])], {"axis": 0})
        is not None
    )

    # ScatterElements
    assert (
        cf._evaluate_node("ScatterElements", [data, np.array([[0, 0], [0, 0]]), data], {"axis": 0})
        is not None
    )

    # NonZero
    assert cf._evaluate_node("NonZero", [np.array([0, 1, 0, 2])], {}) is not None
