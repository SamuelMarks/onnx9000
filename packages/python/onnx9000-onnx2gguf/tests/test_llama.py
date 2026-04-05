"""Tests for llama."""

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Node, Tensor
from onnx9000.onnx2gguf.llama import extract_llama_metadata


def test_extract_llama():
    """Provides functional implementation."""
    g = Graph("llama")
    g.add_tensor(Tensor("model.embed_tokens.weight", (32000, 4096), DType.FLOAT32))
    g.add_tensor(Tensor("model.layers.0.self_attn.q_proj.weight", (4096, 4096), DType.FLOAT32))
    g.add_tensor(Tensor("model.layers.0.self_attn.k_proj.weight", (1024, 4096), DType.FLOAT32))
    g.add_tensor(Tensor("model.layers.0.mlp.up_proj.weight", (11008, 4096), DType.FLOAT32))
    g.add_node(Node("Silu", [], []))
    g.add_node(Node("RMSNormalization", [], [], {"epsilon": 1e-6}))

    meta = extract_llama_metadata(g)
    assert meta["llama.embedding_length"] == 4096
    assert meta["llama.attention.head_count"] == 32
    assert meta["llama.attention.head_count_kv"] == 8
    assert meta["llama.feed_forward_length"] == 11008
    assert meta["llama.vocab_size"] == 32000
    assert meta["llama.block_count"] == 1
    assert meta["custom.is_swiglu"]
    assert meta["llama.attention.layer_norm_rms_epsilon"] == 1e-6
