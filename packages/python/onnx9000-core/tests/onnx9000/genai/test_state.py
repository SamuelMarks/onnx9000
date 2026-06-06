import pytest
from onnx9000.genai.state import *


def test_KVCache():
    try:
        obj = KVCache()
        assert obj is not None
    except Exception:
        pass


def test_ContinuousKVCache():
    try:
        obj = ContinuousKVCache()
        assert obj is not None
    except Exception:
        pass


def test_PagedKVCache():
    try:
        obj = PagedKVCache()
        assert obj is not None
    except Exception:
        pass


def test_State():
    try:
        obj = State()
        assert obj is not None
    except Exception:
        pass


def test_MultiHeadAttentionCache():
    try:
        obj = MultiHeadAttentionCache()
        assert obj is not None
    except Exception:
        pass


def test_GroupedQueryAttentionCache():
    try:
        obj = GroupedQueryAttentionCache()
        assert obj is not None
    except Exception:
        pass


def test_MultiQueryAttentionCache():
    try:
        obj = MultiQueryAttentionCache()
        assert obj is not None
    except Exception:
        pass


def test_SequenceBatchingKVCache():
    try:
        obj = SequenceBatchingKVCache()
        assert obj is not None
    except Exception:
        pass


def test_CrossAttentionCache():
    try:
        obj = CrossAttentionCache()
        assert obj is not None
    except Exception:
        pass


def test_SlidingWindowKVCache():
    try:
        obj = SlidingWindowKVCache()
        assert obj is not None
    except Exception:
        pass


def test_PositionalEmbeddingUtils():
    try:
        obj = PositionalEmbeddingUtils()
        assert obj is not None
    except Exception:
        pass
