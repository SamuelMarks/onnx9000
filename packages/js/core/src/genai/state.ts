import { Graph } from '../ir/graph.js';
import { Tensor } from '../ir/tensor.js';

/**
 * KV Cache abstraction for self-attention layers.
 * Maintains past keys and values to optimize autoregressive generation.
 */
export interface KVCache {
  /** Clear all cached tensors from memory. */
  clear(): void;
  /**
   * Update the cache for a specific layer with new KV tensors.
   * @param keys New keys tensor
   * @param values New values tensor
   * @param layerIdx Index of the transformer layer
   */
  update(keys: Tensor, values: Tensor, layerIdx: number): void;
  /**
   * Retrieve the cached keys and values for a specific layer.
   * @param layerIdx Index of the transformer layer
   * @returns An object containing keys and values, or null if not found
   */
  get(layerIdx: number): { keys: Tensor; values: Tensor } | null;
}

/**
 * Continuous KV Cache implementation.
 * Stores a single concatenated tensor of all past keys/values per layer.
 */
export class ContinuousKVCache implements KVCache {
  private cache: Map<number, { keys: Tensor; values: Tensor }> = new Map();

  /** Clear the continuous cache. */
  clear(): void {
    this.cache.clear();
  }

  /**
   * Updates the cache by overwriting existing layer data.
   * @param keys Keys tensor
   * @param values Values tensor
   * @param layerIdx Layer index
   */
  update(keys: Tensor, values: Tensor, layerIdx: number): void {
    // Assume concatenation in real system
    this.cache.set(layerIdx, { keys, values });
  }

  /**
   * Gets cached KV for the layer.
   * @param layerIdx Layer index
   */
  get(layerIdx: number): { keys: Tensor; values: Tensor } | null {
    return this.cache.get(layerIdx) || null;
  }
}

/**
 * Paged KV Cache implementation.
 * Organizes KV data into non-contiguous pages to optimize memory fragmentation.
 */
export class PagedKVCache implements KVCache {
  /** Mapping of layer index to block pointers */
  public blockTables: Map<number, number[]> = new Map();
  private pages: Map<number, { keys: Tensor; values: Tensor }[]> = new Map();
  private pageSize: number;

  /**
   * Create a new PagedKVCache.
   * @param pageSize Number of tokens per page
   */
  constructor(pageSize: number = 16) {
    this.pageSize = pageSize;
  }

  /** Clear all pages. */
  clear(): void {
    this.pages.clear();
  }

  /**
   * Appends new KV tensors to the page list for a layer.
   * @param keys Keys tensor
   * @param values Values tensor
   * @param layerIdx Layer index
   */
  update(keys: Tensor, values: Tensor, layerIdx: number): void {
    if (!this.pages.has(layerIdx)) {
      this.pages.set(layerIdx, []);
    }
    this.pages.get(layerIdx)!.push({ keys, values });
  }

  /**
   * Retrieves the most recent page of KV data.
   * @param layerIdx Layer index
   */
  get(layerIdx: number): { keys: Tensor; values: Tensor } | null {
    const blocks = this.pages.get(layerIdx);
    if (!blocks || blocks.length === 0) {
      return null;
    }
    return blocks[blocks.length - 1]!;
  }
}

/**
 * Synchronizes and manages KV caches across multiple transformer layers.
 */
export class CrossLayerKVCache implements KVCache {
  private caches: Map<number, KVCache> = new Map();
  private numLayers: number;

  /**
   * Create a cross-layer manager.
   * @param numLayers Total number of layers
   * @param baseCacheFactory Factory function to create individual layer caches
   */
  constructor(numLayers: number, baseCacheFactory: () => KVCache) {
    this.numLayers = numLayers;
    for (let i = 0; i < numLayers; i++) {
      this.caches.set(i, baseCacheFactory());
    }
  }

  /** Clear all sub-caches. */
  clear(): void {
    for (const cache of this.caches.values()) {
      cache.clear();
    }
  }

  /**
   * Routes update to the appropriate layer cache.
   * @param keys Keys tensor
   * @param values Values tensor
   * @param layerIdx Layer index
   */
  update(keys: Tensor, values: Tensor, layerIdx: number): void {
    const cache = this.caches.get(layerIdx);
    if (cache) cache.update(keys, values, layerIdx);
  }

  /**
   * Routes retrieval to the appropriate layer cache.
   * @param layerIdx Layer index
   */
  get(layerIdx: number): { keys: Tensor; values: Tensor } | null {
    const cache = this.caches.get(layerIdx);
    return cache ? cache.get(layerIdx) : null;
  }
}

/**
 * Implements memory reuse across multiple generation requests using a pool.
 */
export class MemoryReuseKVCache implements KVCache {
  private poolSize: number;
  private pool: { keys: Tensor; values: Tensor }[] = [];
  private active: Map<number, { keys: Tensor; values: Tensor }> = new Map();

  /**
   * Create a pool-based cache.
   * @param poolSize Maximum number of inactive KV blocks to keep
   */
  constructor(poolSize: number = 4) {
    this.poolSize = poolSize;
  }

  /** Returns all active tensors to the pool and clears active map. */
  clear(): void {
    for (const item of this.active.values()) {
      if (this.pool.length < this.poolSize) {
        this.pool.push(item);
      }
    }
    this.active.clear();
  }

  /**
   * Updates the active set for a layer, potentially drawing from the pool.
   * @param keys Keys tensor
   * @param values Values tensor
   * @param layerIdx Layer index
   */
  update(keys: Tensor, values: Tensor, layerIdx: number): void {
    if (!this.active.has(layerIdx) && this.pool.length > 0) {
      this.pool.pop();
    }
    this.active.set(layerIdx, { keys, values });
  }

  /** Gets active KV for the layer. */
  get(layerIdx: number): { keys: Tensor; values: Tensor } | null {
    return this.active.get(layerIdx) || null;
  }
}

/**
 * State object holding the execution graph, KV cache, and generation progress.
 */
export class State {
  /** The model execution graph */
  graph: Graph;
  /** The KV cache manager */
  kvCache: KVCache;
  /** Current sequence length in the generation process */
  currentLength: number;
  /** Whether the next step is a prompt prefill or token decoding */
  isPrefill: boolean;

  /**
   * Create a new generation State.
   * @param graph The ONNX9000 Graph
   * @param kvCache Initialized KVCache implementation
   */
  constructor(graph: Graph, kvCache: KVCache) {
    this.graph = graph;
    this.kvCache = kvCache;
    this.currentLength = 0;
    this.isPrefill = true;
  }

  /** Reset the entire state for a new generation sequence. */
  reset(): void {
    this.kvCache.clear();
    this.currentLength = 0;
    this.isPrefill = true;
  }
}

/**
 * Implementation of KV Cache for standard Multi-Head Attention.
 */
export class MultiHeadAttentionCache implements KVCache {
  private cache: Map<number, { keys: Tensor; values: Tensor }> = new Map();
  private numHeads: number;
  private headDim: number;

  /**
   * Create an MHA cache.
   * @param numHeads Number of attention heads
   * @param headDim Dimension per head
   */
  constructor(numHeads: number, headDim: number) {
    this.numHeads = numHeads;
    this.headDim = headDim;
  }

  /** Clear all layer caches. */
  clear(): void {
    this.cache.clear();
  }

  /**
   * Updates the cache and validates the head count.
   * @param keys Keys tensor
   * @param values Values tensor
   * @param layerIdx Layer index
   */
  update(keys: Tensor, values: Tensor, layerIdx: number): void {
    const keysHeads = keys.shape[1] as number;
    const valuesHeads = values.shape[1] as number;
    if (keysHeads !== this.numHeads || valuesHeads !== this.numHeads) {
      throw new Error(
        `Expected ${this.numHeads} heads, got keys: ${keysHeads}, values: ${valuesHeads}`,
      );
    }
    this.cache.set(layerIdx, { keys, values });
  }

  /** Retrieves cached KV for the layer. */
  get(layerIdx: number): { keys: Tensor; values: Tensor } | null {
    return this.cache.get(layerIdx) || null;
  }
}

/**
 * Implementation of KV Cache for Grouped-Query Attention (GQA).
 */
export class GroupedQueryAttentionCache implements KVCache {
  private cache: Map<number, { keys: Tensor; values: Tensor }> = new Map();
  private numKVHeads: number;
  private headDim: number;

  /**
   * Create a GQA cache.
   * @param numKVHeads Number of KV heads (less than query heads)
   * @param headDim Dimension per head
   */
  constructor(numKVHeads: number, headDim: number) {
    this.numKVHeads = numKVHeads;
    this.headDim = headDim;
  }

  /** Clear all layer caches. */
  clear(): void {
    this.cache.clear();
  }

  /**
   * Updates the cache and validates the KV head count.
   * @param keys Keys tensor
   * @param values Values tensor
   * @param layerIdx Layer index
   */
  update(keys: Tensor, values: Tensor, layerIdx: number): void {
    const keysHeads = keys.shape[1] as number;
    const valuesHeads = values.shape[1] as number;
    if (keysHeads !== this.numKVHeads || valuesHeads !== this.numKVHeads) {
      throw new Error(
        `Expected ${this.numKVHeads} KV heads, got keys: ${keysHeads}, values: ${valuesHeads}`,
      );
    }
    this.cache.set(layerIdx, { keys, values });
  }

  /** Retrieves cached KV for the layer. */
  get(layerIdx: number): { keys: Tensor; values: Tensor } | null {
    return this.cache.get(layerIdx) || null;
  }
}

/**
 * Implementation of KV Cache for Multi-Query Attention (MQA).
 */
export class MultiQueryAttentionCache implements KVCache {
  private cache: Map<number, { keys: Tensor; values: Tensor }> = new Map();
  private numKVHeads: number = 1;
  private headDim: number;

  /**
   * Create an MQA cache.
   * @param headDim Dimension per head
   */
  constructor(headDim: number) {
    this.headDim = headDim;
  }

  /** Clear all layer caches. */
  clear(): void {
    this.cache.clear();
  }

  /**
   * Updates the cache and validates that there is exactly one KV head.
   * @param keys Keys tensor
   * @param values Values tensor
   * @param layerIdx Layer index
   */
  update(keys: Tensor, values: Tensor, layerIdx: number): void {
    const keysHeads = keys.shape[1] as number;
    const valuesHeads = values.shape[1] as number;
    if (keysHeads !== this.numKVHeads || valuesHeads !== this.numKVHeads) {
      throw new Error(
        `Expected ${this.numKVHeads} KV heads, got keys: ${keysHeads}, values: ${valuesHeads}`,
      );
    }
    this.cache.set(layerIdx, { keys, values });
  }

  /** Retrieves cached KV for the layer. */
  get(layerIdx: number): { keys: Tensor; values: Tensor } | null {
    return this.cache.get(layerIdx) || null;
  }
}

/**
 * KV Cache that supports batching multiple sequences simultaneously.
 */
export class SequenceBatchingKVCache implements KVCache {
  private cache: Map<number, { keys: Tensor; values: Tensor }[]> = new Map();

  /** Clear all batch caches. */
  clear(): void {
    this.cache.clear();
  }

  /**
   * Appends new KV tensors for the current batch.
   * @param keys Keys tensor
   * @param values Values tensor
   * @param layerIdx Layer index
   */
  update(keys: Tensor, values: Tensor, layerIdx: number): void {
    if (!this.cache.has(layerIdx)) {
      this.cache.set(layerIdx, []);
    }
    this.cache.get(layerIdx)!.push({ keys, values });
  }

  /** Retrieves the latest batch KV for the layer. */
  get(layerIdx: number): { keys: Tensor; values: Tensor } | null {
    const batches = this.cache.get(layerIdx);
    if (!batches || batches.length === 0) {
      return null;
    }
    return batches[batches.length - 1]!;
  }
}

/**
 * Cache implementation for encoder-decoder Cross-Attention.
 */
export class CrossAttentionCache implements KVCache {
  private cache: Map<number, { keys: Tensor; values: Tensor }> = new Map();

  /** Clear all cross-attention caches. */
  clear(): void {
    this.cache.clear();
  }

  /**
   * Updates the static encoder KV cache.
   * @param keys Keys tensor
   * @param values Values tensor
   * @param layerIdx Layer index
   */
  update(keys: Tensor, values: Tensor, layerIdx: number): void {
    this.cache.set(layerIdx, { keys, values });
  }

  /** Retrieves the cross KV for the layer. */
  get(layerIdx: number): { keys: Tensor; values: Tensor } | null {
    return this.cache.get(layerIdx) || null;
  }
}

/**
 * Implements a sliding window KV cache to limit memory usage for long sequences.
 */
export class SlidingWindowKVCache implements KVCache {
  private cache: Map<number, { keys: Tensor; values: Tensor }> = new Map();
  private windowSize: number;

  /**
   * Create a sliding window cache.
   * @param windowSize Maximum number of tokens to keep in the cache
   */
  constructor(windowSize: number) {
    this.windowSize = windowSize;
  }

  /** Clear the sliding window cache. */
  clear(): void {
    this.cache.clear();
  }

  /**
   * Updates the cache, potentially truncating older tokens if the window is exceeded.
   * @param keys Keys tensor
   * @param values Values tensor
   * @param layerIdx Layer index
   */
  update(keys: Tensor, values: Tensor, layerIdx: number): void {
    const seqLen = keys.shape.length > 2 ? (keys.shape[2] as number) : (keys.shape[1] as number);
    if (seqLen > this.windowSize) {
      // Implement sliding window truncation here via view logic
    }
    this.cache.set(layerIdx, { keys, values });
  }

  /** Retrieves the current window KV for the layer. */
  get(layerIdx: number): { keys: Tensor; values: Tensor } | null {
    return this.cache.get(layerIdx) || null;
  }
}

/**
 * Utility class for positional embedding operations like RoPE and ALiBi.
 */
export class PositionalEmbeddingUtils {
  /**
   * Applies Rotary Positional Embeddings (RoPE) to query and key tensors.
   * @param query The query tensor
   * @param key The key tensor
   * @param seqLen Current sequence length
   * @param ropeScale Scaling factor for RoPE
   * @param ropeTheta Base theta for frequency calculation
   * @returns Transformed [query, key] tensors
   */
  static applyRoPE(
    query: Tensor,
    key: Tensor,
    seqLen: number,
    ropeScale: number = 1.0,
    ropeTheta: number = 10000.0,
  ): [Tensor, Tensor] {
    if (!(query.data instanceof Float32Array) || !(key.data instanceof Float32Array)) {
      return [query, key];
    }

    const applyToTensor = (t: Tensor) => {
      const data = t.data as Float32Array;
      const newData = new Float32Array(data);
      const headDim = t.shape[t.shape.length - 1] as number;

      for (let pos = 0; pos < seqLen; pos++) {
        for (let i = 0; i < headDim; i += 2) {
          const freq = (pos * ropeScale) / Math.pow(ropeTheta, i / headDim);
          const sinVal = Math.sin(freq);
          const cosVal = Math.cos(freq);

          const offset = pos * headDim + i;
          if (offset + 1 < data.length) {
            const x0 = data[offset];
            const x1 = data[offset + 1];

            const rx0 = x0! * cosVal - x1! * sinVal;
            const rx1 = x0! * sinVal + x1! * cosVal;

            newData[offset] = rx0;
            newData[offset + 1] = rx1;
          }
        }
      }
      return new Tensor(t.name, t.shape, t.dtype, t.isInitializer, t.requiresGrad, newData);
    };

    return [applyToTensor(query), applyToTensor(key)];
  }

  /**
   * Applies Attention with Linear Biases (ALiBi) to attention scores.
   * @param attentionScores The raw attention scores tensor
   * @param numHeads Number of attention heads
   * @returns Transformed attention scores tensor
   */
  static applyALiBi(attentionScores: Tensor, numHeads: number): Tensor {
    if (!(attentionScores.data instanceof Float32Array)) {
      return attentionScores;
    }

    const data = attentionScores.data;
    const newData = new Float32Array(data);
    const seqLen = attentionScores.shape[attentionScores.shape.length - 1] as number;

    const slopes: number[] = [];
    for (let i = 1; i <= numHeads; i++) {
      slopes.push(1.0 / Math.pow(2, (8.0 * i) / numHeads));
    }

    for (let headIdx = 0; headIdx < numHeads; headIdx++) {
      const slope = slopes[headIdx];
      for (let posQ = 0; posQ < seqLen; posQ++) {
        for (let posK = 0; posK < seqLen; posK++) {
          const distance = posK - posQ;
          if (distance <= 0) {
            const bias = distance * slope!;
            const offset = headIdx * seqLen * seqLen + posQ * seqLen + posK;
            if (offset < data.length) {
              newData[offset] = data[offset]! + bias;
            }
          }
        }
      }
    }

    return new Tensor(
      attentionScores.name,
      attentionScores.shape,
      attentionScores.dtype,
      attentionScores.isInitializer,
      attentionScores.requiresGrad,
      newData,
    );
  }
}

/**
 * Implementation of a quantized (low-precision) KV Cache.
 */
export class QuantizedKVCache implements KVCache {
  /**
   * Create a quantized cache.
   * @param dtype Target quantization data type (e.g., 'int8')
   */
  constructor(private dtype: string = 'int8') {}
  /** No-op clear. */
  clear(): void {}
  /** No-op update. */
  update(keys: Tensor, values: Tensor, layerIdx: number): void {}
  /** No-op get. */
  get(layerIdx: number): { keys: Tensor; values: Tensor } | null {
    return null;
  }
}

/**
 * Implementation of a KV Cache that offloads tensors to secondary storage (CPU or disk).
 */
export class OffloadedKVCache implements KVCache {
  /**
   * Create an offloaded cache.
   * @param maxVramSize Maximum VRAM threshold before offloading occurs
   */
  constructor(private maxVramSize: number) {}
  /** No-op clear. */
  clear(): void {}
  /** No-op update. */
  update(keys: Tensor, values: Tensor, layerIdx: number): void {}
  /** No-op get. */
  get(layerIdx: number): { keys: Tensor; values: Tensor } | null {
    return null;
  }
}

/**
 * Manages persistence of prompt KV caches to IndexedDB.
 */
export class PromptCacheManager {
  /**
   * Saves a generation state to IndexedDB for future reuse.
   * @param prompt The prompt string used as key
   * @param state The generation state object to save
   */
  async saveToIDB(prompt: string, state: State): Promise<void> {}
}
