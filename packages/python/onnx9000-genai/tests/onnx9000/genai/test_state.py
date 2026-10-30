"""Unit tests for KV-cache and positional embeddings in `onnx9000.genai.state`."""

import struct

from onnx9000.core.dtypes import DType
from onnx9000.core.ir import Graph, Tensor
from onnx9000.genai.state import (
    ContinuousKVCache,
    CrossAttentionCache,
    GroupedQueryAttentionCache,
    KVCache,
    MultiHeadAttentionCache,
    MultiQueryAttentionCache,
    PagedKVCache,
    PositionalEmbeddingUtils,
    SequenceBatchingKVCache,
    SlidingWindowKVCache,
    State,
)


def test_kv_caches_and_positional_embeddings() -> None:
    """Verify KVCache, ContinuousKVCache, PagedKVCache, State, MHA/GQA/MQA caches, SequenceBatching, CrossAttention, SlidingWindow, and PositionalEmbeddingUtils."""
    base_kv = KVCache()
    assert base_kv.get(0) is None

    ckv = ContinuousKVCache()
    k = Tensor(name="k", shape=(1, 2, 1, 4), dtype=DType.FLOAT32, data=bytearray(32))
    v = Tensor(name="v", shape=(1, 2, 1, 4), dtype=DType.FLOAT32, data=bytearray(32))
    ckv.update(k, v, 0)
    assert ckv.get(0) is not None

    pkv = PagedKVCache(page_size=4)
    pkv.update(k, v, 0)
    assert pkv.get(0) == (k, v)
    pkv.clear()
    assert pkv.get(0) is None

    st = State(Graph("st_g"), ckv)
    st.reset()
    assert st.current_length == 0 and st.is_prefill is True

    mha = MultiHeadAttentionCache(num_heads=2, head_dim=4)
    mha.update(k, v, 0)
    assert mha.get(0) == (k, v)

    gqa = GroupedQueryAttentionCache(num_kv_heads=2, head_dim=4)
    gqa.update(k, v, 0)
    assert gqa.get(0) == (k, v)

    k1 = Tensor(name="k1", shape=(1, 1, 1, 4), dtype=DType.FLOAT32, data=bytearray(16))
    v1 = Tensor(name="v1", shape=(1, 1, 1, 4), dtype=DType.FLOAT32, data=bytearray(16))
    mqa = MultiQueryAttentionCache(head_dim=4)
    mqa.update(k1, v1, 0)
    assert mqa.get(0) == (k1, v1)

    sb = SequenceBatchingKVCache()
    sb.update(k, v, 0)
    assert sb.get(0) == (k, v)

    ca = CrossAttentionCache()
    ca.update(k, v, 0)
    assert ca.get(0) == (k, v)

    sw = SlidingWindowKVCache(window_size=4)
    sw.update(k, v, 0)
    assert sw.get(0) == (k, v)

    q_rope, k_rope = PositionalEmbeddingUtils.apply_rope(k1, v1, seq_len=1)
    assert q_rope.shape == (1, 1, 1, 4) and k_rope.shape == (1, 1, 1, 4)

    scores = Tensor(
        name="scores",
        shape=(1, 1, 2, 2),
        dtype=DType.FLOAT32,
        data=bytearray(struct.pack("<4f", 1.0, 1.0, 1.0, 1.0)),
    )
    alibi_out = PositionalEmbeddingUtils.apply_alibi(scores, num_heads=1)
    assert alibi_out.shape == (1, 1, 2, 2)
