"""Coverage gaps for simplifier constant folding."""

import pytest
import numpy as np
import logging
from onnx9000.core.ir import Graph, Node, Constant, Attribute, Tensor
from onnx9000.core.dtypes import DType
from onnx9000.optimizer.simplifier.passes.constant_folding import ConstantFoldingPass


def test_reduce_ops_constant_folding_all():
    """Verify constant folding for all Reduce* operators using initializers."""
    ops = [
        "ReduceSum",
        "ReduceMean",
        "ReduceMax",
        "ReduceMin",
        "ReduceProd",
        "ReduceL1",
        "ReduceL2",
        "ReduceLogSum",
        "ReduceLogSumExp",
    ]

    cf_pass = ConstantFoldingPass()
    for op in ops:
        graph = Graph(f"test_{op}")
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        c = Constant("input", data.tobytes(), (2, 2), DType.FLOAT32)
        graph.add_tensor(c)
        graph.initializers.append("input")

        n = Node(op, ["input"], ["output"], attributes={"axes": [1], "keepdims": 0})
        graph.add_node(n)

        cf_pass.run(graph)
        assert any(node.op_type == "Constant" and "output" in node.outputs for node in graph.nodes)


def test_reduce_ops_axes_input():
    """Verify constant folding for Reduce* with axes as input using initializers."""
    graph = Graph("test_axes_input")
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    axes = np.array([0], dtype=np.int64)

    c = Constant("input", data.tobytes(), (2, 2), DType.FLOAT32)
    graph.add_tensor(c)
    graph.initializers.append("input")

    a = Constant("axes", axes.tobytes(), (1,), DType.INT64)
    graph.add_tensor(a)
    graph.initializers.append("axes")

    n = Node("ReduceSum", ["input", "axes"], ["output"], attributes={"keepdims": 1})
    graph.add_node(n)

    cf_pass = ConstantFoldingPass()
    cf_pass.run(graph)
    assert any(node.op_type == "Constant" and "output" in node.outputs for node in graph.nodes)


def test_constant_folding_logger_gaps(caplog):
    """Verify logger warning/info branches in constant_folding.py."""
    with caplog.at_level(logging.INFO):
        graph = Graph("test_logger")
        data = np.array([1.0], dtype=np.float32)
        c = Constant("input", data.tobytes(), (1,), DType.FLOAT32)
        graph.add_tensor(c)
        graph.initializers.append("input")

        n = Node("Relu", ["input"], ["output"])
        graph.add_node(n)

        cf_pass = ConstantFoldingPass()
        cf_pass.run(graph)
        assert "Folded node" in caplog.text


def test_constant_folding_unsupported_domain(caplog):
    """Verify warning for unsupported CustomOp domain (line 361)."""
    with caplog.at_level(logging.WARNING):
        graph = Graph("test_domain")
        n = Node("CustomOp", [], ["out"], domain="com.microsoft")
        graph.add_node(n)

        cf_pass = ConstantFoldingPass()
        cf_pass.run(graph)
        assert "unsupported CustomOp domain" in caplog.text
