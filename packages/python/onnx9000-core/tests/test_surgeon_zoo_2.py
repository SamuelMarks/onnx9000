import pytest
from onnx9000.core.ir import Attribute, Graph, Node
from onnx9000.core.surgeon import (
    LayoutOptimizerPass,
    StatefulToStatelessPass,
    map_fft,
    map_while_loop,
    unroll_scan,
)


def test_unroll_scan():
    graph = Graph(name="test")
    graph.nodes = []
    attr = Attribute(name="sequence_length", attr_type="int", value=3)
    node = Node(
        op_type="jax.lax.scan",
        name="scan1",
        inputs=["in1"],
        outputs=["out1"],
        domain="",
        attributes={"sequence_length": attr},
    )
    graph.nodes.append(node)

    graph = unroll_scan(graph)
    assert len(graph.nodes) == 3
    assert graph.nodes[0].op_type == "ScanUnrolled"
    assert graph.nodes[0].outputs == ["out1_0"]
    assert graph.nodes[2].outputs == ["out1_2"]


def test_map_while_loop():
    graph = Graph(name="test")
    graph.nodes = []
    node = Node(
        op_type="jax.lax.while_loop",
        name="w1",
        inputs=["state1"],
        outputs=[],
        domain="",
        attributes={},
    )
    graph.nodes.append(node)

    graph = map_while_loop(graph)
    assert graph.nodes[0].op_type == "Loop"
    assert graph.nodes[0].inputs == ["trip_count_var", "cond_var", "state1"]


def test_stateful_to_stateless():
    graph = Graph(name="test")
    graph.nodes = []
    graph.inputs = []
    graph.outputs = []
    node = Node(
        op_type="MambaBlock",
        name="mamba1",
        inputs=["in1"],
        outputs=["out1"],
        domain="",
        attributes={},
    )
    graph.nodes.append(node)

    graph = StatefulToStatelessPass.apply(graph)
    assert "mamba1_state_in" in graph.nodes[0].inputs
    assert "mamba1_state_out" in graph.nodes[0].outputs
    assert "mamba1_state_in" in graph.inputs
    assert "mamba1_state_out" in graph.outputs


def test_map_fft():
    graph = Graph(name="test")
    graph.nodes = []
    node = Node(
        op_type="jnp.fft.rfft2", name="fft1", inputs=[], outputs=[], domain="", attributes={}
    )
    graph.nodes.append(node)

    graph = map_fft(graph)
    assert graph.nodes[0].op_type == "DFT"
    assert graph.nodes[0].attributes["inverse"].value == 0
    assert graph.nodes[0].attributes["spatial"].value == 1


def test_layout_optimizer():
    graph = Graph(name="test")
    graph.nodes = []
    node = Node(op_type="Conv", name="conv1", inputs=[], outputs=[], domain="", attributes={})
    graph.nodes.append(node)

    graph = LayoutOptimizerPass.apply(graph)
    assert graph.nodes[0].attributes["layout"].value == "NCHW"
