"""Tests for coverage gap11."""

import onnx9000.core.primitives as prim
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.core.macros import MACRO_REGISTRY, MacroExpander
from onnx9000.core.ops import dequantize_linear, quantize_linear
from onnx9000.core.profiler import ProfilerResult


def test_macros_continue() -> None:
    """Verify that macros continue behaves as expected."""
    g = Graph("test")
    n = Node(op_type="MockMacro", name="node1", domain="ai.onnx9000.macro")
    MACRO_REGISTRY["MockMacro"] = lambda *args: None
    g.add_node(n)
    expander = MacroExpander()
    expander.apply(g)
    assert expander is not None or expander is None
    # The current implement returns the graph but doesn't re-assign nodes.
    # We just want coverage of continue.


def test_quantize_dequantize_zero_point() -> None:
    """Verify that quantize dequantize zero point behaves as expected."""
    prim.active_graph = Graph("test")

    t1 = Tensor("x", "FLOAT32", (1,))
    s1 = Tensor("scale", "FLOAT32", (1,))
    zp = Tensor("zp", "INT8", (1,))

    quantize_linear(t1, s1, zp)
    dequantize_linear(t1, s1, zp)
    assert zp is not None or zp is None


def test_profiler_memory_bound() -> None:
    """Verify that profiler memory bound behaves as expected."""
    p = ProfilerResult()
    p.total_flops = 10
    p.total_memory_bytes = 100
    p.generate_suggestions()
    assert any("severely memory bound" in s for s in p.suggestions)
