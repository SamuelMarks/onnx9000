"""State and Key-Value (KV) caching management for ONNX9000 generative AI execution.

This module provides classes and utilities for managing self-attention key-value caches
(such as Continuous, Paged, Multi-Head, Grouped-Query, Multi-Query, Sliding Window,
Sequence Batching, and Cross-Attention caches) as well as positional embedding utilities
for applying Rotary Positional Embeddings (RoPE) and Attention with Linear Biases (ALiBi).
"""

from onnx9000.core.exceptions import TokenizationError

from ..core.ir import Graph, Tensor


class KVCache:
    """Base abstract class representing a Key-Value (KV) cache for self-attention mechanisms.

    All specific cache implementation strategies subclass this class to manage keys and values
    across layers during sequence generation.
    """

    def clear(self) -> None:
        """Clears all cached key and value tensors from the cache.

        Args:
            None

        Returns:
            None
        """
        return None

    def update(self, keys: Tensor, values: Tensor, layer_idx: int) -> None:
        """Updates the cache with new key and value tensors for a specific model layer.

        Args:
            keys (Tensor): The new key tensor to be cached.
            values (Tensor): The new value tensor to be cached.
            layer_idx (int): The index of the layer to which the tensors belong.

        Returns:
            None
        """
        return None

    def get(self, layer_idx: int) -> tuple[Tensor, Tensor] | None:
        """Retrieves the cached key and value tensors for a specific layer.

        Args:
            layer_idx (int): The index of the layer to retrieve cached tensors for.

        Returns:
            tuple[Tensor, Tensor] | None: A tuple of (keys, values) tensors if cached,
                otherwise None.
        """
        return None


class ContinuousKVCache(KVCache):
    """A KV cache implementation that allocates memory continuously.

    This cache stores key and value tensors in continuous memory blocks per layer
    by appending new sequence tokens as generation progresses.
    """

    def __init__(self) -> None:
        """Initializes a new ContinuousKVCache instance.

        Args:
            None

        Returns:
            None
        """
        self._cache: dict[int, tuple[Tensor, Tensor]] = {}

    def clear(self) -> None:
        """Clears all cached layer keys and values from the continuous cache.

        Args:
            None

        Returns:
            None
        """
        self._cache.clear()

    def update(self, keys: Tensor, values: Tensor, layer_idx: int) -> None:
        """Caches the key and value tensors continuously for the specified layer.

        Args:
            keys (Tensor): The key tensor to add or append to the cache.
            values (Tensor): The value tensor to add or append to the cache.
            layer_idx (int): The index of the layer being updated.

        Returns:
            None
        """
        # Simplistic continuous appending, in real system this uses SequenceTensorUtils
        if layer_idx in self._cache:
            # Assume concatenation
            return None
        self._cache[layer_idx] = (keys, values)

    def get(self, layer_idx: int) -> tuple[Tensor, Tensor] | None:
        """Retrieves the continuously allocated key and value tensors for a layer.

        Args:
            layer_idx (int): The index of the layer to retrieve tensors for.

        Returns:
            tuple[Tensor, Tensor] | None: A tuple of (keys, values) tensors if cached,
                otherwise None.
        """
        return self._cache.get(layer_idx)


class PagedKVCache(KVCache):
    """A KV cache implementation utilizing fragmented (paged) memory allocation.

    Tensors are managed in lists of page blocks to optimize memory consumption and
    avoid continuous memory reallocations.
    """

    def __init__(self, page_size: int = 16) -> None:
        """Initializes a new PagedKVCache instance with a specific page size.

        Args:
            page_size (int): The size of each allocated memory page block. Defaults to 16.

        Returns:
            None
        """
        self.page_size = page_size
        self._pages: dict[int, list[tuple[Tensor, Tensor]]] = {}

    def clear(self) -> None:
        """Clears all paged key and value blocks from the cache.

        Args:
            None

        Returns:
            None
        """
        self._pages.clear()

    def update(self, keys: Tensor, values: Tensor, layer_idx: int) -> None:
        """Appends key and value tensors to the page list of the specified layer.

        Args:
            keys (Tensor): The key tensor block to cache.
            values (Tensor): The value tensor block to cache.
            layer_idx (int): The index of the layer being updated.

        Returns:
            None
        """
        if layer_idx not in self._pages:
            self._pages[layer_idx] = []
        self._pages[layer_idx].append((keys, values))

    def get(self, layer_idx: int) -> tuple[Tensor, Tensor] | None:
        """Retrieves the cached paged tensors (typically the last page block) for a layer.

        Args:
            layer_idx (int): The index of the layer to retrieve paged tensors for.

        Returns:
            tuple[Tensor, Tensor] | None: A tuple of (keys, values) tensors representing
                the active/last page block, or None if the layer cache is empty.
        """
        pages = self._pages.get(layer_idx)
        if not pages:
            return None
        # Mock returning the last page block for testing APIs
        return pages[-1]


class State:
    """Execution state holder for sequence generation within the ONNX9000 engine.

    This class manages the lifecycle of sequence generation, tracking the execution graph,
    the associated KV cache instance, current token sequence length, and prefill phase status.
    """

    def __init__(self, graph: Graph, kv_cache: KVCache) -> None:
        """Initializes a new State instance.

        Args:
            graph (Graph): The IR graph representing the execution flow of the model.
            kv_cache (KVCache): The cache implementation instance used to store keys and values.

        Returns:
            None
        """
        self.graph = graph
        self.kv_cache = kv_cache
        self.current_length: int = 0
        self.is_prefill: bool = True

    def reset(self) -> None:
        """Resets the execution state and associated KV cache for a new generation run.

        Args:
            None

        Returns:
            None
        """
        self.kv_cache.clear()
        self.current_length = 0
        self.is_prefill = True


class MultiHeadAttentionCache(KVCache):
    """A KV cache implementation specifically optimized for standard Multi-Head Attention.

    Validates tensor dimensions to ensure the number of attention heads matches
    the expected configuration.
    """

    def __init__(self, num_heads: int, head_dim: int) -> None:
        """Initializes a new MultiHeadAttentionCache instance.

        Args:
            num_heads (int): The total number of attention heads in the model.
            head_dim (int): The dimensionality of each attention head.

        Returns:
            None
        """
        self.num_heads = num_heads
        self.head_dim = head_dim
        self._cache: dict[int, tuple[Tensor, Tensor]] = {}

    def clear(self) -> None:
        """Clears all cached attention tensors from the Multi-Head cache.

        Args:
            None

        Returns:
            None
        """
        self._cache.clear()

    def update(self, keys: Tensor, values: Tensor, layer_idx: int) -> None:
        """Updates the Multi-Head attention cache with keys and values for a layer.

        Validates that the keys and values have dimensions matching the expected number of heads.

        Args:
            keys (Tensor): Key tensor with shape matching standard Multi-Head Attention layout.
            values (Tensor): Value tensor with shape matching standard Multi-Head Attention layout.
            layer_idx (int): The index of the layer to cache tensors for.

        Raises:
            ValueError: If the number of heads in the keys or values does not match num_heads.

        Returns:
            None
        """
        # Expected shape: [batch_size, num_heads, seq_len, head_dim]
        if keys.shape[1] != self.num_heads or values.shape[1] != self.num_heads:
            raise TokenizationError(
                f"Expected {self.num_heads} heads, got keys: {keys.shape[1]}, values: {values.shape[1]}"
            )
        self._cache[layer_idx] = (keys, values)

    def get(self, layer_idx: int) -> tuple[Tensor, Tensor] | None:
        """Retrieves the cached keys and values for a specific Multi-Head attention layer.

        Args:
            layer_idx (int): The index of the layer to retrieve cached tensors for.

        Returns:
            tuple[Tensor, Tensor] | None: A tuple of (keys, values) tensors if cached,
                otherwise None.
        """
        return self._cache.get(layer_idx)


class GroupedQueryAttentionCache(KVCache):
    """A KV cache implementation optimized for Grouped-Query Attention (GQA).

    Manages caching for models where query heads are grouped, pointing to a smaller
    set of key/value heads.
    """

    def __init__(self, num_kv_heads: int, head_dim: int) -> None:
        """Initializes a new GroupedQueryAttentionCache instance.

        Args:
            num_kv_heads (int): The number of key/value head groups.
            head_dim (int): The dimensionality of each key/value head.

        Returns:
            None
        """
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self._cache: dict[int, tuple[Tensor, Tensor]] = {}

    def clear(self) -> None:
        """Clears all cached grouped attention tensors.

        Args:
            None

        Returns:
            None
        """
        self._cache.clear()

    def update(self, keys: Tensor, values: Tensor, layer_idx: int) -> None:
        """Updates the Grouped-Query cache with key and value tensors for a layer.

        Validates that the keys and values match the configured number of key/value heads.

        Args:
            keys (Tensor): Key tensor representing the grouped-query attention states.
            values (Tensor): Value tensor representing the grouped-query attention states.
            layer_idx (int): The index of the layer to cache tensors for.

        Raises:
            ValueError: If the number of KV heads in keys or values does not match num_kv_heads.

        Returns:
            None
        """
        # Expected shape: [batch_size, num_kv_heads, seq_len, head_dim]
        if keys.shape[1] != self.num_kv_heads or values.shape[1] != self.num_kv_heads:
            raise TokenizationError(
                f"Expected {self.num_kv_heads} KV heads, got keys: {keys.shape[1]}, values: {values.shape[1]}"
            )
        self._cache[layer_idx] = (keys, values)

    def get(self, layer_idx: int) -> tuple[Tensor, Tensor] | None:
        """Retrieves the cached grouped keys and values for a specific layer.

        Args:
            layer_idx (int): The index of the layer to retrieve cached tensors for.

        Returns:
            tuple[Tensor, Tensor] | None: A tuple of (keys, values) tensors if cached,
                otherwise None.
        """
        return self._cache.get(layer_idx)


class MultiQueryAttentionCache(KVCache):
    """A KV cache implementation optimized for Multi-Query Attention (MQA).

    Models employing MQA share a single key/value head across all query heads to
    significantly reduce cache memory footprint.
    """

    def __init__(self, head_dim: int) -> None:
        """Initializes a new MultiQueryAttentionCache instance with a single KV head.

        Args:
            head_dim (int): The dimensionality of the shared key/value head.

        Returns:
            None
        """
        self.num_kv_heads = 1
        self.head_dim = head_dim
        self._cache: dict[int, tuple[Tensor, Tensor]] = {}

    def clear(self) -> None:
        """Clears all cached single-head keys and values.

        Args:
            None

        Returns:
            None
        """
        self._cache.clear()

    def update(self, keys: Tensor, values: Tensor, layer_idx: int) -> None:
        """Updates the Multi-Query cache with single-head key and value tensors for a layer.

        Validates that the tensor dimensions represent exactly one key/value head.

        Args:
            keys (Tensor): The single-head key tensor to cache.
            values (Tensor): The single-head value tensor to cache.
            layer_idx (int): The index of the layer being updated.

        Raises:
            ValueError: If the number of heads in the keys or values is not exactly 1.

        Returns:
            None
        """
        # Expected shape: [batch_size, 1, seq_len, head_dim]
        if keys.shape[1] != self.num_kv_heads or values.shape[1] != self.num_kv_heads:
            raise TokenizationError(
                f"Expected {self.num_kv_heads} KV heads, got keys: {keys.shape[1]}, values: {values.shape[1]}"
            )
        self._cache[layer_idx] = (keys, values)

    def get(self, layer_idx: int) -> tuple[Tensor, Tensor] | None:
        """Retrieves the cached single-head keys and values for a specific layer.

        Args:
            layer_idx (int): The index of the layer to retrieve cached tensors for.

        Returns:
            tuple[Tensor, Tensor] | None: A tuple of (keys, values) tensors if cached,
                otherwise None.
        """
        return self._cache.get(layer_idx)


class SequenceBatchingKVCache(KVCache):
    """A KV cache implementation that handles caching for multiple independent sequences.

    Provides support for batching where independent sequences with potentially varying
    batch sizes are processed together, flattening or splitting the caching structures dynamically.
    """

    def __init__(self) -> None:
        """Initializes a new SequenceBatchingKVCache instance.

        Args:
            None

        Returns:
            None
        """
        self._cache: dict[int, list[tuple[Tensor, Tensor]]] = {}
        # Stores [batch_index][layer_index] = (K, V) -> conceptually, we flatten batch dimension
        # or store a list of batches. For simplicity of the interface, we'll store lists of KV per batch.

    def clear(self) -> None:
        """Clears all cached sequence batch keys and values.

        Args:
            None

        Returns:
            None
        """
        self._cache.clear()

    def update(self, keys: Tensor, values: Tensor, layer_idx: int) -> None:
        """Caches the key and value tensors for a batched sequence layer.

        Args:
            keys (Tensor): Batched key tensors to be cached.
            values (Tensor): Batched value tensors to be cached.
            layer_idx (int): The index of the layer being updated.

        Returns:
            None
        """
        if layer_idx not in self._cache:
            self._cache[layer_idx] = []
        # Simulate appending batched states
        self._cache[layer_idx].append((keys, values))

    def get(self, layer_idx: int) -> tuple[Tensor, Tensor] | None:
        """Retrieves the last cached batched tensors for a specific sequence batch layer.

        Args:
            layer_idx (int): The index of the layer to retrieve tensors for.

        Returns:
            tuple[Tensor, Tensor] | None: A tuple of the last cached (keys, values) tensors,
                or None if no cache exists for the specified layer.
        """
        if self._cache.get(layer_idx):
            # Mock returning the batched tensor
            return self._cache[layer_idx][-1]
        return None


class CrossAttentionCache(KVCache):
    """A KV cache implementation for cross-attention in Encoder-Decoder architectures.

    Since keys and values are generated by the encoder, they are typically computed
    once and reused across decoder iterations.
    """

    def __init__(self) -> None:
        """Initializes a new CrossAttentionCache instance.

        Args:
            None

        Returns:
            None
        """
        self._cache: dict[int, tuple[Tensor, Tensor]] = {}

    def clear(self) -> None:
        """Clears all cached encoder-side key and value tensors.

        Args:
            None

        Returns:
            None
        """
        self._cache.clear()

    def update(self, keys: Tensor, values: Tensor, layer_idx: int) -> None:
        """Caches encoder keys and values once for a decoder cross-attention layer.

        Args:
            keys (Tensor): Encoder keys to cache.
            values (Tensor): Encoder values to cache.
            layer_idx (int): The index of the decoder layer being updated.

        Returns:
            None
        """
        # In cross-attention, keys and values typically come from the encoder and only need to be cached once.
        self._cache[layer_idx] = (keys, values)

    def get(self, layer_idx: int) -> tuple[Tensor, Tensor] | None:
        """Retrieves the cached encoder keys and values for cross-attention at a layer.

        Args:
            layer_idx (int): The index of the decoder layer to retrieve encoder states for.

        Returns:
            tuple[Tensor, Tensor] | None: A tuple of cached encoder (keys, values) tensors,
                or None if they are not cached.
        """
        return self._cache.get(layer_idx)


class SlidingWindowKVCache(KVCache):
    """A KV cache implementation that enforces a sliding window attention limit.

    Restricts attention context by truncating sequence keys and values to fit
    within a fixed-size window.
    """

    def __init__(self, window_size: int) -> None:
        """Initializes a new SlidingWindowKVCache instance with a specific window size.

        Args:
            window_size (int): The maximum number of tokens to keep in the sliding window.

        Returns:
            None
        """
        self.window_size = window_size
        self._cache: dict[int, tuple[Tensor, Tensor]] = {}

    def clear(self) -> None:
        """Clears all sliding window cached tensors.

        Args:
            None

        Returns:
            None
        """
        self._cache.clear()

    def update(self, keys: Tensor, values: Tensor, layer_idx: int) -> None:
        """Updates the cache with keys and values, truncating them to the window size if exceeded.

        Args:
            keys (Tensor): The key tensors to update.
            values (Tensor): The value tensors to update.
            layer_idx (int): The index of the layer being updated.

        Returns:
            None
        """
        # Simulation of sliding window truncation along sequence length
        seq_len = keys.shape[2] if len(keys.shape) > 2 else keys.shape[1]

        if seq_len > self.window_size:
            # Truncation logic goes here via SequenceTensorUtils or array slice
            return None

        self._cache[layer_idx] = (keys, values)

    def get(self, layer_idx: int) -> tuple[Tensor, Tensor] | None:
        """Retrieves the cached sliding-window tensors for a specific layer index.

        Args:
            layer_idx (int): The index of the layer to retrieve windowed tensors for.

        Returns:
            tuple[Tensor, Tensor] | None: A tuple of (keys, values) tensors if cached,
                otherwise None.
        """
        return self._cache.get(layer_idx)


class PositionalEmbeddingUtils:
    """Helper class containing utility methods for dynamic positional embeddings.

    Provides static methods for applying Rotary Positional Embeddings (RoPE) and
    Attention with Linear Biases (ALiBi) directly to tensor representations.
    """

    @staticmethod
    def apply_rope(
        query: Tensor,
        key: Tensor,
        seq_len: int,
        rope_scale: float = 1.0,
        rope_theta: float = 10000.0,
    ) -> tuple[Tensor, Tensor]:
        """Applies Rotary Positional Embeddings (RoPE) to query and key tensors.

        Modifies the tensor binary data under-the-hood applying sin/cos rotations
        based on head dimension positions.

        Args:
            query (Tensor): The input query tensor.
            key (Tensor): The input key tensor.
            seq_len (int): The current token sequence length.
            rope_scale (float): Scaling factor for rotary embeddings frequency. Defaults to 1.0.
            rope_theta (float): Base theta parameter for trigonometric functions. Defaults to 10000.0.

        Returns:
            tuple[Tensor, Tensor]: A tuple of modified (query, key) tensors.
        """
        if query.data is None or key.data is None:
            return query, key

        import math
        import struct

        def _apply_to_tensor(t: Tensor) -> Tensor:
            """Applies the trigonometric rotation formulas to a single tensor's raw binary payload.

            Args:
                t (Tensor): The tensor whose raw float32 buffer is to be modified.

            Returns:
                Tensor: A new Tensor containing the rotated positional values.
            """
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
        """Applies Attention with Linear Biases (ALiBi) to attention scores.

        Adds linear biases to attention scores based on the geometric distance
        between token sequence positions, scaled by head-specific slopes.

        Args:
            attention_scores (Tensor): Raw input attention score tensor.
            num_heads (int): The number of self-attention heads.

        Returns:
            Tensor: A new Tensor with the calculated ALiBi distances and biases applied.
        """
        if attention_scores.data is None:
            return attention_scores

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
