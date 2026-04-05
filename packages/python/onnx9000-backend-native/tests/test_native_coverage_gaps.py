"""Tests for missing coverage gaps in onnx9000-backend-native.

This module provides tests to ensure 100% code coverage for the native backend.
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from onnx9000.backends.codegen.compiler import compile_cpp
from onnx9000.backends.cpu.ops import OP_REGISTRY
from onnx9000.backends.tensorrt.provider import TensorrtExecutionProvider
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Attribute, Node


def test_cpu_ops_coverage_gaps() -> None:
    """Test missing lines in cpu/ops.py.

    Covers Gelu, Reduce ops, Transpose, Flatten, Slice, Mod, BitShift,
    Cast, Activation ops, Squeeze, Unsqueeze, Unique, Compress,
    Trilu, and ConcatFromSequence.
    """
    # gelu_op
    gelu = OP_REGISTRY["Gelu"]
    a = np.array([1.0, 2.0], dtype=np.float32)
    gelu([a], {})

    # reducesum_op with no axes and keepdims=0
    reducesum = OP_REGISTRY["ReduceSum"]
    reducesum([a], {"keepdims": 0})

    # reducemean_op with no axes and keepdims=0
    reducemean = OP_REGISTRY["ReduceMean"]
    reducemean([a], {"keepdims": 0})

    # reducemax_op
    reducemax = OP_REGISTRY["ReduceMax"]
    reducemax([a], {"keepdims": 0})

    # transpose_op with perm
    transpose = OP_REGISTRY["Transpose"]
    b = np.ones((2, 3), dtype=np.float32)
    transpose([b], {"perm": [1, 0]})

    # flatten_op with axis < 0
    flatten = OP_REGISTRY["Flatten"]
    flatten([b], {"axis": -1})

    # slice_op with steps and more than 3 inputs
    slice_op = OP_REGISTRY["Slice"]
    data = np.ones((10, 10), dtype=np.float32)
    starts = np.array([0, 0], dtype=np.int64)
    ends = np.array([5, 5], dtype=np.int64)
    axes = np.array([0, 1], dtype=np.int64)
    steps = np.array([2, 2], dtype=np.int64)
    slice_op([data, starts, ends, axes, steps], {})

    # mod_op (fmod=0)
    mod = OP_REGISTRY["Mod"]
    mod([a, a], {"fmod": 0})

    # bitshift_op (direction="RIGHT")
    bitshift = OP_REGISTRY["BitShift"]
    a_int = np.array([4, 8], dtype=np.int32)
    bitshift([a_int, np.array([1, 1], dtype=np.int32)], {"direction": "RIGHT"})

    # reducel1_op
    reducel1 = OP_REGISTRY["ReduceL1"]
    reducel1([a], {"keepdims": 0})

    # reducel2_op
    reducel2 = OP_REGISTRY["ReduceL2"]
    reducel2([a], {"keepdims": 0})

    # reducelogsum_op
    reducelogsum = OP_REGISTRY["ReduceLogSum"]
    reducelogsum([a], {"keepdims": 0})

    # reducelogsumexp_op
    reducelogsumexp = OP_REGISTRY["ReduceLogSumExp"]
    reducelogsumexp([a], {"axes": []})

    # reducesumsquare_op
    reducesumsquare = OP_REGISTRY["ReduceSumSquare"]
    reducesumsquare([a], {"keepdims": 0})

    # einsum_op
    einsum = OP_REGISTRY["Einsum"]
    einsum([a, a], {"equation": "i,i->"})

    # cast_op
    cast = OP_REGISTRY["Cast"]
    for to_type in [1, 2, 3, 4, 5, 6, 7, 9, 11]:
        cast([a], {"to": to_type})

    # leakyrelu_op
    leakyrelu = OP_REGISTRY["LeakyRelu"]
    leakyrelu([a], {"alpha": 0.1})

    # prelu_op
    prelu = OP_REGISTRY["PRelu"]
    prelu([a, a], {})

    # elu_op
    elu = OP_REGISTRY["Elu"]
    elu([a], {"alpha": 1.0})

    # selu_op
    selu = OP_REGISTRY["Selu"]
    selu([a], {})

    # hardsigmoid_op
    hardsigmoid = OP_REGISTRY["HardSigmoid"]
    hardsigmoid([a], {})

    # logsoftmax_op
    logsoftmax = OP_REGISTRY["LogSoftmax"]
    logsoftmax([a], {})

    # hardmax_op
    hardmax = OP_REGISTRY["Hardmax"]
    hardmax([b], {"axis": 1})

    # squeeze_op with axes
    squeeze = OP_REGISTRY["Squeeze"]
    c = np.ones((1, 2, 1), dtype=np.float32)
    squeeze([c, np.array([0, 2], dtype=np.int64)], {})

    # unsqueeze_op
    unsqueeze = OP_REGISTRY["Unsqueeze"]
    unsqueeze([a, np.array([0], dtype=np.int64)], {})

    # unique_op
    unique = OP_REGISTRY["Unique"]
    unique([a], {})

    # compress_op with axis
    compress = OP_REGISTRY["Compress"]
    compress([a, np.array([1, 0], dtype=np.bool_)], {"axis": 0})

    # trilu_op (upper=1)
    trilu = OP_REGISTRY["Trilu"]
    trilu([data], {"upper": 1})

    # concatfromsequence_op (new_axis=0)
    concatfromsequence = OP_REGISTRY["ConcatFromSequence"]
    concatfromsequence([[a, a]], {"new_axis": 0})


def test_tensorrt_provider() -> None:
    """Test TensorrtExecutionProvider."""
    provider = TensorrtExecutionProvider()
    graph = MagicMock()
    partitioned = provider.partition_graph(graph)
    assert partitioned == [graph]

    session = MagicMock()
    inputs = {"input": np.ones((1, 3), dtype=np.float32)}
    outputs = provider.run(session, inputs)
    assert "input" in outputs
    assert np.array_equal(outputs["input"], inputs["input"])


def test_codegen_compiler_missing_pybind() -> None:
    """Test compile_cpp when pybind11 is missing."""
    with patch.dict("sys.modules", {"pybind11": None}):
        res = compile_cpp("int main() {}", use_pybind=True)
        assert res is None


def test_codegen_ops_nn_coverage_gaps() -> None:
    """Test missing lines in codegen/ops/nn.py."""
    from onnx9000.backends.codegen.ops.nn import generate_batchnorm, generate_transpose

    # Mock generator context
    ctx = MagicMock()
    ctx.get_tensor_name.side_effect = lambda x: x
    ctx.tensor_offsets = {"out": 0, "out_bn": 0}

    # Mock tensor info
    tensor_in = MagicMock()
    tensor_in.dtype = DType.FLOAT32
    tensor_in.shape = [1, 2]
    tensor_in.buffer_id = 0
    tensor_in.is_initializer = False

    tensor_out = MagicMock()
    tensor_out.dtype = DType.FLOAT32
    tensor_out.shape = [2, 1]
    tensor_out.buffer_id = 1
    tensor_out.is_initializer = False

    ctx.graph.tensors = {"in": tensor_in, "out": tensor_out}
    ctx.graph.inputs = []

    # 88-89: Transpose with perm
    node_transpose = Node(
        op_type="Transpose",
        inputs=["in"],
        outputs=["out"],
        attributes={"perm": Attribute("perm", value=[1, 0])},
    )
    code_transpose = generate_transpose(node_transpose, ctx)
    assert "{1, 0}" in code_transpose

    # 745-753: BatchNormalization
    ctx.graph.tensors.update(
        {
            "x": MagicMock(
                dtype=DType.FLOAT32, shape=[1, 2, 3, 3], buffer_id=2, is_initializer=False
            ),
            "scale": MagicMock(dtype=DType.FLOAT32, shape=[2], buffer_id=3, is_initializer=True),
            "b": MagicMock(dtype=DType.FLOAT32, shape=[2], buffer_id=4, is_initializer=True),
            "mean": MagicMock(dtype=DType.FLOAT32, shape=[2], buffer_id=5, is_initializer=True),
            "var": MagicMock(dtype=DType.FLOAT32, shape=[2], buffer_id=6, is_initializer=True),
            "out_bn": MagicMock(
                dtype=DType.FLOAT32, shape=[1, 2, 3, 3], buffer_id=7, is_initializer=False
            ),
        }
    )
    node_bn = Node(
        op_type="BatchNormalization",
        inputs=["x", "scale", "b", "mean", "var"],
        outputs=["out_bn"],
        attributes={"epsilon": Attribute("epsilon", value=1e-4)},
    )
    code_bn = generate_batchnorm(node_bn, ctx)
    assert "BatchNormalization" in code_bn


def test_codegen_ops_tensor_ops_coverage_gaps() -> None:
    """Test missing lines in codegen/ops/tensor_ops.py."""
    from onnx9000.backends.codegen.ops.tensor_ops import generate_categorymapper

    ctx = MagicMock()
    ctx.get_tensor_name.side_effect = lambda x: x
    ctx.tensor_offsets = {"out": 0}
    ctx.graph.tensors = {
        "in": MagicMock(dtype=DType.INT64, shape=[5], buffer_id=0, is_initializer=False),
        "out": MagicMock(dtype=DType.INT64, shape=[5], buffer_id=1, is_initializer=False),
    }

    # 1220-1224: CategoryMapper with cats_int64s but no cats_strings
    node_cm = Node(
        op_type="CategoryMapper",
        inputs=["in"],
        outputs=["out"],
        attributes={
            "cats_int64s": Attribute("cats_int64s", value=[1, 2, 3]),
            "default_int64": Attribute("default_int64", value=-1),
        },
    )
    code_cm = generate_categorymapper(node_cm, ctx)
    assert code_cm is not None
    assert "CategoryMapper statically generated switch" in code_cm
