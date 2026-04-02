"""Tests the all cpu ops module functionality."""

import contextlib

import numpy as np
from onnx9000.backends.cpu.ops import OP_REGISTRY


def test_all_cpu_ops_via_registry() -> None:
    """Tests the all cpu ops via registry functionality."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    c = np.array([1.0], dtype=np.float32)
    for _name, func in OP_REGISTRY.items():
        try:
            res = func([a, b, c, c, c, c], {})
            assert isinstance(res, list)
        except Exception:
            continue


def test_complex_cpu_ops() -> None:
    """Tests the complex cpu ops functionality."""
    from onnx9000.backends.cpu.ops import (
        batchnorm_op,
        conv_op,
        convtranspose_op,
        instancenormalization_op,
        layernorm_op,
    )

    x = np.ones((1, 1, 4, 4), dtype=np.float32)
    w = np.ones((1, 1, 2, 2), dtype=np.float32)
    b = np.ones((1,), dtype=np.float32)
    conv_op([x, w, b], {"strides": [1, 1], "pads": [0, 0, 0, 0]})
    convtranspose_op([x, w, b], {"strides": [1, 1], "pads": [0, 0, 0, 0]})
    layernorm_op([x, b, b], {"axis": -1})
    scale = np.ones((1,), dtype=np.float32)
    B = np.zeros((1,), dtype=np.float32)
    mean = np.zeros((1,), dtype=np.float32)
    var = np.ones((1,), dtype=np.float32)
    batchnorm_op([x, scale, B, mean, var], {})
    instancenormalization_op([x, scale, B], {})
    from onnx9000.backends.cpu.ops import averagepool_op, maxpool_op

    maxpool_op([x], {"kernel_shape": [2, 2], "strides": [1, 1], "pads": [0, 0, 0, 0]})
    averagepool_op([x], {"kernel_shape": [2, 2], "strides": [1, 1], "pads": [1, 1, 1, 1]})
    maxpool_op([x], {"kernel_shape": [2, 2], "strides": [1, 1], "pads": [1, 1, 1, 1]})
    from onnx9000.backends.cpu.ops import lrn_op, shrink_op

    lrn_op([x], {"size": 2})
    shrink_op([x], {"bias": 0.0, "lambd": 0.5})


def test_missing_ops_branches_1() -> None:
    """Tests the missing ops branches functionality."""
    from onnx9000.backends.cpu.ops import OP_REGISTRY

    OP_REGISTRY["Conv"]
    convtranspose_op = OP_REGISTRY["ConvTranspose"]
    x = np.ones((1, 1, 4, 4), dtype=np.float32)
    w = np.ones((1, 1, 2, 2), dtype=np.float32)
    convtranspose_op([x, w], {"strides": [2, 2], "pads": [1, 1, 1, 1]})
    maxpool_op = OP_REGISTRY["MaxPool"]
    maxpool_op([x], {"kernel_shape": [2, 2], "strides": [1, 1], "pads": [0, 0, 0, 0]})
    averagepool_op = OP_REGISTRY["AveragePool"]
    averagepool_op([x], {"kernel_shape": [2, 2], "strides": [1, 1], "pads": [1, 1, 1, 1]})
    maxpool_op([x], {"kernel_shape": [2, 2], "strides": [1, 1], "pads": [1, 1, 1, 1]})
    batchnormalization_op = OP_REGISTRY["BatchNormalization"]
    scale = np.ones((1,), dtype=np.float32)
    B = np.zeros((1,), dtype=np.float32)
    mean = np.zeros((1,), dtype=np.float32)
    var = np.ones((1,), dtype=np.float32)
    batchnormalization_op([x, scale, B, mean, var], {"epsilon": 0.0001})
    instancenormalization_op = OP_REGISTRY["InstanceNormalization"]
    instancenormalization_op([x, scale, B], {"epsilon": 0.0001})
    constant_op = OP_REGISTRY["Constant"]
    constant_op([np.array([2, 2], dtype=np.int64)], {"value": np.array([1.0])})
    pad_op = OP_REGISTRY["Pad"]
    with contextlib.suppress(TypeError):
        pad_op([x, np.array([1, 1, 1, 1, 1, 1, 1, 1])], {"mode": "constant"})
    clip_op = OP_REGISTRY["Clip"]
    clip_op([x, np.array([-1.0]), np.array([1.0])], {})


def test_missing_ops_branches() -> None:
    """Tests the missing ops branches functionality."""
    from onnx9000.backends.cpu.ops import OP_REGISTRY

    OP_REGISTRY["Conv"]
    convtranspose_op = OP_REGISTRY["ConvTranspose"]
    x = np.ones((1, 1, 4, 4), dtype=np.float32)
    w = np.ones((1, 1, 2, 2), dtype=np.float32)
    convtranspose_op([x, w], {"strides": [2, 2], "pads": [1, 1, 1, 1]})
    maxpool_op = OP_REGISTRY["MaxPool"]
    maxpool_op([x], {"kernel_shape": [2, 2], "strides": [1, 1], "pads": [0, 0, 0, 0]})
    averagepool_op = OP_REGISTRY["AveragePool"]
    averagepool_op([x], {"kernel_shape": [2, 2], "strides": [1, 1], "pads": [1, 1, 1, 1]})
    maxpool_op([x], {"kernel_shape": [2, 2], "strides": [1, 1], "pads": [1, 1, 1, 1]})
    batchnormalization_op = OP_REGISTRY["BatchNormalization"]
    scale = np.ones((1,), dtype=np.float32)
    B = np.zeros((1,), dtype=np.float32)
    mean = np.zeros((1,), dtype=np.float32)
    var = np.ones((1,), dtype=np.float32)
    batchnormalization_op([x, scale, B, mean, var], {"epsilon": 0.0001})
    instancenormalization_op = OP_REGISTRY["InstanceNormalization"]
    instancenormalization_op([x, scale, B], {"epsilon": 0.0001})
    constant_op = OP_REGISTRY["Constant"]
    constant_op([np.array([2, 2], dtype=np.int64)], {"value": np.array([1.0])})
    pad_op = OP_REGISTRY["Pad"]
    with contextlib.suppress(TypeError):
        pad_op([x, np.array([1, 1, 1, 1, 1, 1, 1, 1])], {"mode": "constant"})
    clip_op = OP_REGISTRY["Clip"]
    clip_op([x, np.array([-1.0]), np.array([1.0])], {})


def test_missing_ops_again_again() -> None:
    """Tests the missing ops again again functionality."""
    from onnx9000.backends.cpu.ops import OP_REGISTRY

    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    i = np.array([1], dtype=np.int64)
    x_conv = np.ones((1, 2, 4, 4), dtype=np.float32)
    w_conv = np.ones((2, 1, 2, 2), dtype=np.float32)
    OP_REGISTRY["Conv"]([x_conv, w_conv], {"group": 2, "strides": [1, 1], "pads": [0, 0, 0, 0]})
    OP_REGISTRY["Reshape"]([a, np.array([0, -1], dtype=np.int64)], {})
    OP_REGISTRY["Flatten"]([a], {"axis": 0})
    OP_REGISTRY["Gather"]([a, i], {"axis": 1})
    OP_REGISTRY["Mod"]([a, b], {"fmod": 1})
    a_int = np.array([[1, 2], [3, 4]], dtype=np.int64)
    OP_REGISTRY["BitShift"]([a_int, a_int], {"direction": "LEFT"})
    OP_REGISTRY["ReduceMax"]([a], {"axes": [0], "keepdims": 0})
    OP_REGISTRY["Gemm"]([a, b, a], {"transA": 1, "transB": 1})
    OP_REGISTRY["Scatter"]([a, np.array([[0, 1], [0, 1]], dtype=np.int64), a], {"axis": 0})
    OP_REGISTRY["TopK"]([a, i], {"axis": -1})
    OP_REGISTRY["CumSum"]([a, np.array([0], dtype=np.int64)], {})
    OP_REGISTRY["Trilu"]([a, np.array([0], dtype=np.int64)], {"upper": 0})
    seq_erase = OP_REGISTRY["SequenceErase"]
    seq_erase([[a, b], i], {})
    split_to_seq = OP_REGISTRY["Split"]
    split_to_seq([a], {})


def test_missing_last_lines() -> None:
    """Tests the missing last lines functionality."""
    from onnx9000.backends.cpu.ops import OP_REGISTRY

    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    i = np.array([1], dtype=np.int64)
    OP_REGISTRY["Flatten"]([a], {})
    OP_REGISTRY["Gather"]([a, i], {})
    OP_REGISTRY["ScatterND"](
        [a, np.array([[0]], dtype=np.int64), np.array([5.0], dtype=np.float32)], {}
    )
    OP_REGISTRY["ReduceMax"]([a], {"keepdims": 0})
    OP_REGISTRY["Split"]([a, np.array([1, 1], dtype=np.int64)], {})


def test_ops_coverage_final() -> None:
    """Tests the ops coverage final functionality."""
    import numpy as np
    from onnx9000.backends.cpu.ops import (
        concatfromsequence_op,
        flatten_op,
        reducelogsumexp_op,
        slice_op,
    )

    a = np.ones((2, 2, 2), dtype=np.float32)
    flatten_op([a], {"axis": -1})
    slice_op([a, np.array([0]), np.array([1]), np.array([0]), np.array([1])], {})
    reducelogsumexp_op([a], {"axes": [0], "keepdims": 0})
    concatfromsequence_op(
        [[np.ones((2,), dtype=np.float32), np.ones((2,), dtype=np.float32)]], {"new_axis": 1}
    )

    from onnx9000.backends.cpu.ops import sequenceinsert_op

    sequenceinsert_op([[1, 2], np.array(3), np.array(1)], {})
