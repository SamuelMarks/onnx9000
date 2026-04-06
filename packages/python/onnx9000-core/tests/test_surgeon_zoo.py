import pytest
from onnx9000.core.ir import Attribute, Graph, Node
from onnx9000.core.surgeon import (
    map_alibi,
    map_bonsai_rope,
    map_flash_attention,
    map_gqa_mqa,
    normalize_sliding_window_attention,
)


def test_map_bonsai_rope():
    graph = Graph(name="test")
    graph.nodes = []
    node = Node(op_type="BonsaiRoPE", name="rope1", inputs=[], outputs=[], domain="", attributes={})
    graph.nodes.append(node)

    graph = map_bonsai_rope(graph)
    assert graph.nodes[0].op_type == "RotaryEmbedding"


def test_map_alibi():
    graph = Graph(name="test")
    graph.nodes = []
    node = Node(
        op_type="ALiBi", name="alibi1", inputs=["in1"], outputs=["out1"], domain="", attributes={}
    )
    graph.nodes.append(node)

    graph = map_alibi(graph)
    assert len(graph.nodes) == 2
    assert graph.nodes[0].op_type == "Add"
    assert graph.nodes[0].inputs == ["in1"]
    assert graph.nodes[0].outputs == ["alibi_add_out"]
    assert graph.nodes[1].op_type == "Mask"
    assert graph.nodes[1].inputs == ["alibi_add_out"]
    assert graph.nodes[1].outputs == ["out1"]


def test_map_gqa_mqa():
    graph = Graph(name="test")
    graph.nodes = []
    node1 = Node(op_type="MQA", name="m1", inputs=[], outputs=[], domain="", attributes={})
    node2 = Node(op_type="GQA", name="m2", inputs=[], outputs=[], domain="", attributes={})
    graph.nodes.extend([node1, node2])

    graph = map_gqa_mqa(graph)
    assert graph.nodes[0].op_type == "MultiHeadAttention"
    assert graph.nodes[0].attributes["kv_num_heads"].value == 1
    assert graph.nodes[1].op_type == "MultiHeadAttention"
    assert graph.nodes[1].attributes["kv_num_heads"].value == 8


def test_normalize_sliding_window_attention():
    graph = Graph(name="test")
    graph.nodes = []
    node = Node(
        op_type="SlidingWindowAttention", name="s1", inputs=[], outputs=[], domain="", attributes={}
    )
    graph.nodes.append(node)

    graph = normalize_sliding_window_attention(graph)
    assert graph.nodes[0].op_type == "MultiHeadAttention"
    assert graph.nodes[0].attributes["causal_masking"].value == 1
    assert graph.nodes[0].attributes["window_size"].value == 4096


def test_map_flash_attention():
    graph = Graph(name="test")
    graph.nodes = []
    node1 = Node(
        op_type="jax.lax.pallas_call",
        name="my_flash_attn_call",
        inputs=[],
        outputs=[],
        domain="",
        attributes={},
    )
    node2 = Node(
        op_type="jax.lax.pallas_call",
        name="other_call",
        inputs=[],
        outputs=[],
        domain="",
        attributes={},
    )
    graph.nodes.extend([node1, node2])

    graph = map_flash_attention(graph)
    assert graph.nodes[0].op_type == "FlashAttention"
    assert graph.nodes[1].op_type == "jax.lax.pallas_call"
