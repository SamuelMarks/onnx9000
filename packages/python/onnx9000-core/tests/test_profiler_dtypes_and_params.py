"""Tests the profiler dtypes and params module functionality."""

import pytest
from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.core.profiler import profile


def test_profiler_dtypes():
    """Tests the profiler dtypes functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("x16", [10, 10], DType.FLOAT16))
    g.add_tensor(Tensor("w16", [10, 10], DType.FLOAT16, is_initializer=True))
    g.initializers.append("w16")
    g.inputs.append("x16")
    g.add_node(Node("MatMul", ["x16", "w16"], ["y16"]))
    g.add_tensor(Tensor("y16", [10, 10], DType.FLOAT16))

    g.add_tensor(Tensor("x8", [10, 10], DType.INT8))
    g.add_tensor(Tensor("w8", [10, 10], DType.INT8, is_initializer=True))
    g.initializers.append("w8")
    g.inputs.append("x8")
    g.add_node(Node("MatMul", ["x8", "w8"], ["y8"]))
    g.add_tensor(Tensor("y8", [10, 10], DType.INT8))

    g.add_tensor(Tensor("xb", [10, 10], DType.BFLOAT16))
    g.add_tensor(Tensor("wb", [10, 10], DType.BFLOAT16, is_initializer=True))
    g.initializers.append("wb")
    g.inputs.append("xb")
    g.add_node(Node("MatMul", ["xb", "wb"], ["yb"]))
    g.add_tensor(Tensor("yb", [10, 10], DType.BFLOAT16))

    res = profile(g)
    assert res.fp16_macs > 0
    assert res.int8_macs > 0


def test_profiler_batchnorm_int():
    """Tests the profiler batchnorm int functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("x", [10, 10], DType.FLOAT32))
    g.inputs.append("x")
    g.add_node(Node("BatchNormalization", ["x"], ["y"]))
    # mock to not erase shape
    import onnx9000.core.profiler

    orig = onnx9000.core.profiler.infer_shapes_and_types
    onnx9000.core.profiler.infer_shapes_and_types = lambda x: None
    g.add_tensor(Tensor("y", [10, 10], DType.FLOAT32))
    res = profile(g)
    onnx9000.core.profiler.infer_shapes_and_types = orig

    assert res.total_flops == 400


def test_profiler_reduce_int():
    """Tests the profiler reduce int functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("x", [10, 10], DType.FLOAT32))
    g.inputs.append("x")
    g.add_node(Node("ReduceMean", ["x"], ["y"]))

    import onnx9000.core.profiler

    orig = onnx9000.core.profiler.infer_shapes_and_types
    onnx9000.core.profiler.infer_shapes_and_types = lambda x: None
    g.add_tensor(Tensor("y", [10, 10], DType.FLOAT32))
    res = profile(g)
    onnx9000.core.profiler.infer_shapes_and_types = orig

    assert res.total_flops > 0


def test_profiler_batchnorm_str():
    """Tests the profiler batchnorm str functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("x", ("B", "I"), DType.FLOAT32))
    g.inputs.append("x")
    g.add_node(Node("BatchNormalization", ["x"], ["y"]))

    import onnx9000.core.profiler

    orig = onnx9000.core.profiler.infer_shapes_and_types
    onnx9000.core.profiler.infer_shapes_and_types = lambda x: None
    g.add_tensor(Tensor("y", ("B", "I"), DType.FLOAT32))
    res = profile(g)
    onnx9000.core.profiler.infer_shapes_and_types = orig

    assert "4 *" in str(res.total_flops)


def test_profiler_prelu():
    """Tests the profiler prelu functionality."""
    g = Graph("g")
    g.add_tensor(Tensor("x", [10, 10], DType.FLOAT32))
    g.add_node(Node("PRelu", ["x"], ["y"]))
    g.add_tensor(Tensor("y", [10, 10], DType.FLOAT32))
    res = profile(g)
    assert res.total_flops > 0


from onnx9000.core.profiler import _add_metric


def test_add_metric():
    """Tests the add metric functionality."""
    assert _add_metric(0, "A") == "A"
    assert _add_metric("B", 0) == "B"
    assert _add_metric("A", "B") == "(A + B)"
