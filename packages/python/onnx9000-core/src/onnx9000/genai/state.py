from typing import Dict, List, Optional, Tuple

from ..core.ir import Graph, Tensor


class KVCache:
    """KV Cache abstraction for self-attention."""

    def clear(self) -> None:
        pass

    def update(self, keys: Tensor, values: Tensor, layer_idx: int) -> None:
        pass

    def get(self, layer_idx: int) -> Optional[tuple[Tensor, Tensor]]:
        return None


class ContinuousKVCache(KVCache):
    """Implements continuous memory allocation for caches."""

    def __init__(self):
        self._cache: dict[int, tuple[Tensor, Tensor]] = {}

    def clear(self) -> None:
        self._cache.clear()

    def update(self, keys: Tensor, values: Tensor, layer_idx: int) -> None:
        # Simplistic continuous appending, in real system this uses SequenceTensorUtils
        if layer_idx in self._cache:
            # Assume concatenation
            pass
        self._cache[layer_idx] = (keys, values)

    def get(self, layer_idx: int) -> Optional[tuple[Tensor, Tensor]]:
        return self._cache.get(layer_idx)


class PagedKVCache(KVCache):
    """Implements fragmented (paged) memory allocation for caches."""

    def __init__(self, page_size: int = 16):
        self.page_size = page_size
        self._pages: dict[int, list[tuple[Tensor, Tensor]]] = {}

    def clear(self) -> None:
        self._pages.clear()

    def update(self, keys: Tensor, values: Tensor, layer_idx: int) -> None:
        if layer_idx not in self._pages:
            self._pages[layer_idx] = []
        self._pages[layer_idx].append((keys, values))

    def get(self, layer_idx: int) -> Optional[tuple[Tensor, Tensor]]:
        pages = self._pages.get(layer_idx)
        if not pages:
            return None
        # Mock returning the last page block for testing APIs
        return pages[-1]


class State:
    """State object to hold execution graph and KV cache."""

    def __init__(self, graph: Graph, kv_cache: KVCache):
        self.graph = graph
        self.kv_cache = kv_cache
        self.current_length: int = 0
        self.is_prefill: bool = True

    def reset(self) -> None:
        """Reset state for a new generation"""
        self.kv_cache.clear()
        self.current_length = 0
        self.is_prefill = True


class MultiHeadAttentionCache(KVCache):
    """Supports standard Multi-Head Attention (MHA) caching."""

    def __init__(self, num_heads: int, head_dim: int):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self._cache: dict[int, tuple[Tensor, Tensor]] = {}

    def clear(self) -> None:
        self._cache.clear()

    def update(self, keys: Tensor, values: Tensor, layer_idx: int) -> None:
        # Expected shape: [batch_size, num_heads, seq_len, head_dim]
        if keys.shape[1] != self.num_heads or values.shape[1] != self.num_heads:
            raise ValueError(
                f"Expected {self.num_heads} heads, got keys: {keys.shape[1]}, values: {values.shape[1]}"
            )
        self._cache[layer_idx] = (keys, values)

    def get(self, layer_idx: int) -> Optional[tuple[Tensor, Tensor]]:
        return self._cache.get(layer_idx)


class GroupedQueryAttentionCache(KVCache):
    """Supports Grouped-Query Attention (GQA) caching structures."""

    def __init__(self, num_kv_heads: int, head_dim: int):
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self._cache: dict[int, tuple[Tensor, Tensor]] = {}

    def clear(self) -> None:
        self._cache.clear()

    def update(self, keys: Tensor, values: Tensor, layer_idx: int) -> None:
        # Expected shape: [batch_size, num_kv_heads, seq_len, head_dim]
        if keys.shape[1] != self.num_kv_heads or values.shape[1] != self.num_kv_heads:
            raise ValueError(
                f"Expected {self.num_kv_heads} KV heads, got keys: {keys.shape[1]}, values: {values.shape[1]}"
            )
        self._cache[layer_idx] = (keys, values)

    def get(self, layer_idx: int) -> Optional[tuple[Tensor, Tensor]]:
        return self._cache.get(layer_idx)


class MultiQueryAttentionCache(KVCache):
    """Supports Multi-Query Attention (MQA) caching structures (single KV head)."""

    def __init__(self, head_dim: int):
        self.num_kv_heads = 1
        self.head_dim = head_dim
        self._cache: dict[int, tuple[Tensor, Tensor]] = {}

    def clear(self) -> None:
        self._cache.clear()

    def update(self, keys: Tensor, values: Tensor, layer_idx: int) -> None:
        # Expected shape: [batch_size, 1, seq_len, head_dim]
        if keys.shape[1] != self.num_kv_heads or values.shape[1] != self.num_kv_heads:
            raise ValueError(
                f"Expected {self.num_kv_heads} KV heads, got keys: {keys.shape[1]}, values: {values.shape[1]}"
            )
        self._cache[layer_idx] = (keys, values)

    def get(self, layer_idx: int) -> Optional[tuple[Tensor, Tensor]]:
        return self._cache.get(layer_idx)


class SequenceBatchingKVCache(KVCache):
    """Handles caching for multiple independent sequences with potentially varying batch sizes."""

    def __init__(self):
        self._cache: dict[int, list[tuple[Tensor, Tensor]]] = {}
        # Stores [batch_index][layer_index] = (K, V) -> conceptually, we flatten batch dimension
        # or store a list of batches. For simplicity of the interface, we'll store lists of KV per batch.

    def clear(self) -> None:
        self._cache.clear()

    def update(self, keys: Tensor, values: Tensor, layer_idx: int) -> None:
        if layer_idx not in self._cache:
            self._cache[layer_idx] = []
        # Simulate appending batched states
        self._cache[layer_idx].append((keys, values))

    def get(self, layer_idx: int) -> Optional[tuple[Tensor, Tensor]]:
        if layer_idx in self._cache and self._cache[layer_idx]:
            # Mock returning the batched tensor
            return self._cache[layer_idx][-1]
        return None


class CrossAttentionCache(KVCache):
    """Supports cross-attention caching for Encoder-Decoder architectures."""

    def __init__(self):
        self._cache: dict[int, tuple[Tensor, Tensor]] = {}

    def clear(self) -> None:
        self._cache.clear()

    def update(self, keys: Tensor, values: Tensor, layer_idx: int) -> None:
        # In cross-attention, keys and values typically come from the encoder and only need to be cached once.
        self._cache[layer_idx] = (keys, values)

    def get(self, layer_idx: int) -> Optional[tuple[Tensor, Tensor]]:
        return self._cache.get(layer_idx)


class SlidingWindowKVCache(KVCache):
    """Supports sliding window attention limits (e.g., Mistral)."""

    def __init__(self, window_size: int):
        self.window_size = window_size
        self._cache: dict[int, tuple[Tensor, Tensor]] = {}

    def clear(self) -> None:
        self._cache.clear()

    def update(self, keys: Tensor, values: Tensor, layer_idx: int) -> None:
        import struct

        # Simulation of sliding window truncation along sequence length
        seq_len = keys.shape[2] if len(keys.shape) > 2 else keys.shape[1]

        if seq_len > self.window_size:
            # Truncation logic goes here via SequenceTensorUtils or array slice
            pass

        self._cache[layer_idx] = (keys, values)

    def get(self, layer_idx: int) -> Optional[tuple[Tensor, Tensor]]:
        return self._cache.get(layer_idx)


class PositionalEmbeddingUtils:
    """Utilities for dynamic positional embeddings like RoPE and ALiBi."""

    @staticmethod
    def apply_rope(
        query: Tensor,
        key: Tensor,
        seq_len: int,
        rope_scale: float = 1.0,
        rope_theta: float = 10000.0,
    ) -> tuple[Tensor, Tensor]:
        """Applies Rotary Positional Embeddings to queries and keys."""
        if query.data is None or key.data is None:
            return query, key

        import math
        import struct

        def _apply_to_tensor(t: Tensor) -> Tensor:
            itemsize = t.dtype.itemsize if hasattr(t.dtype, "itemsize") else 4
            new_data = bytearray(t.data)

            # Simplified RoPE logic applied per head per sequence position
            # Real implementation maps native struct buffer mathematically
            head_dim = t.shape[-1]
            for pos in range(seq_len):
                for i in range(0, head_dim, 2):
                    freq = (pos * rope_scale) / (rope_theta ** (i / head_dim))
                    sin_val = math.sin(freq)
                    cos_val = math.cos(freq)

                    # Assuming shape [batch, heads, seq, head_dim]
                    offset = (pos * head_dim + i) * itemsize
                    if offset + itemsize * 2 <= len(new_data):
                        x0 = struct.unpack("<f", t.data[offset : offset + itemsize])[0]
                        x1 = struct.unpack("<f", t.data[offset + itemsize : offset + itemsize * 2])[
                            0
                        ]

                        rx0 = x0 * cos_val - x1 * sin_val
                        rx1 = x0 * sin_val + x1 * cos_val

                        new_data[offset : offset + itemsize] = struct.pack("<f", rx0)
                        new_data[offset + itemsize : offset + itemsize * 2] = struct.pack("<f", rx1)
            return Tensor(name=t.name, shape=t.shape, dtype=t.dtype, data=new_data)

        return _apply_to_tensor(query), _apply_to_tensor(key)

    @staticmethod
    def apply_alibi(attention_scores: Tensor, num_heads: int) -> Tensor:
        """Applies ALiBi (Attention with Linear Biases) to attention scores."""
        if attention_scores.data is None:
            return attention_scores

        import math
        import struct

        itemsize = (
            attention_scores.dtype.itemsize if hasattr(attention_scores.dtype, "itemsize") else 4
        )
        new_data = bytearray(attention_scores.data)

        # ALiBi calculates slopes based on heads
        slopes = [1.0 / (2 ** ((8.0 * i) / num_heads)) for i in range(1, num_heads + 1)]

        seq_len = attention_scores.shape[-1]

        # Simplified ALiBi logic applied to scores
        for head_idx in range(num_heads):
            slope = slopes[head_idx]
            for pos_q in range(seq_len):
                for pos_k in range(seq_len):
                    distance = pos_k - pos_q
                    if distance <= 0:
                        bias = distance * slope

                        # Assuming shape [batch, heads, seq_q, seq_k]
                        offset = (head_idx * seq_len * seq_len + pos_q * seq_len + pos_k) * itemsize
                        if offset + itemsize <= len(new_data):
                            val = struct.unpack(
                                "<f", attention_scores.data[offset : offset + itemsize]
                            )[0]
                            new_data[offset : offset + itemsize] = struct.pack("<f", val + bias)

        return Tensor(
            name=attention_scores.name,
            shape=attention_scores.shape,
            dtype=attention_scores.dtype,
            data=new_data,
        )
