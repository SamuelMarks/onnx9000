import pytest
from onnx9000.core.ir import Graph, Tensor, Node, DynamicDim
from onnx9000.core.dtypes import DType
from onnx9000.optimize.hardware.layout import LayoutOptimizer


def test_chunk_large_tensors_inner_else():
    g = Graph("test")
    t_large_axis0_1 = Tensor("large_1", (1, 1000), DType.FLOAT32)
    g.add_tensor(t_large_axis0_1)
    g2 = LayoutOptimizer.chunk_large_tensors_pass(g, max_size=100)
    assert "large_1" in g2.tensors


def test_layout_nchw_to_nhwc_else():
    g = Graph("test")
    n = Node("Add", ["in1"], ["out1"], {}, "add1")
    g.add_node(n)
    g2 = LayoutOptimizer.nchw_to_nhwc_pass(g)
    assert len(g2.nodes) == 1
    assert g2.nodes[0].op_type == "Add"


def test_layout_nhwc_to_nchw_else():
    g = Graph("test")
    n = Node("Add", ["in1"], ["out1"], {}, "add1")
    g.add_node(n)
    g2 = LayoutOptimizer.nhwc_to_nchw_pass(g)
    assert len(g2.nodes) == 1
    assert g2.nodes[0].op_type == "Add"
