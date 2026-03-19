import { Graph } from '../ir/graph.js';
import { Tensor } from '../ir/tensor.js';

/**
 * KV Cache abstraction for self-attention.
 */
export interface KVCache {
  /** Clear the cache */
  clear(): void;
  /** Update cache with new keys and values */
  update(keys: Tensor, values: Tensor, layerIdx: number): void;
  /** Retrieve past keys and values */
  get(layerIdx: number): { keys: Tensor; values: Tensor } | null;
}

/**
 * Continuous KV Cache implementation
 */
export class ContinuousKVCache implements KVCache {
  private cache: Map<number, { keys: Tensor; values: Tensor }> = new Map();

  clear(): void {
    this.cache.clear();
  }

  update(keys: Tensor, values: Tensor, layerIdx: number): void {
    // Assume concatenation in real system
    this.cache.set(layerIdx, { keys, values });
  }

  get(layerIdx: number): { keys: Tensor; values: Tensor } | null {
    return this.cache.get(layerIdx) || null;
  }
}

/**
 * Paged KV Cache implementation
 */
export class PagedKVCache implements KVCache {
  public blockTables: Map<number, number[]> = new Map(); // WGSL conceptual pointer
  private pages: Map<number, { keys: Tensor; values: Tensor }[]> = new Map();
  private pageSize: number;

  constructor(pageSize: number = 16) {
    this.pageSize = pageSize;
  }

  clear(): void {
    this.pages.clear();
  }

  update(keys: Tensor, values: Tensor, layerIdx: number): void {
    if (!this.pages.has(layerIdx)) {
      this.pages.set(layerIdx, []);
    }
    this.pages.get(layerIdx)!.push({ keys, values });
  }

  get(layerIdx: number): { keys: Tensor; values: Tensor } | null {
    const blocks = this.pages.get(layerIdx);
    if (!blocks || blocks.length === 0) {
      return null;
    }
    return blocks[blocks.length - 1]!;
  }
}

/**
 * State object to hold execution graph and KV cache.
 */

/** Synchronizes KV caches across multiple transformer layers */
export class CrossLayerKVCache implements KVCache {
  private caches: Map<number, KVCache> = new Map();
  private numLayers: number;

  constructor(numLayers: number, baseCacheFactory: () => KVCache) {
    this.numLayers = numLayers;
    for (let i = 0; i < numLayers; i++) {
      this.caches.set(i, baseCacheFactory());
    }
  }

  clear(): void {
    for (const cache of this.caches.values()) {
      cache.clear();
    }
  }

  update(keys: Tensor, values: Tensor, layerIdx: number): void {
    const cache = this.caches.get(layerIdx);
    if (cache) cache.update(keys, values, layerIdx);
  }

  get(layerIdx: number): { keys: Tensor; values: Tensor } | null {
    const cache = this.caches.get(layerIdx);
    return cache ? cache.get(layerIdx) : null;
  }
}

/** Implements memory reuse across generation requests */
export class MemoryReuseKVCache implements KVCache {
  private poolSize: number;
  private pool: { keys: Tensor; values: Tensor }[] = [];
  private active: Map<number, { keys: Tensor; values: Tensor }> = new Map();

  constructor(poolSize: number = 4) {
    this.poolSize = poolSize;
  }

  clear(): void {
    for (const item of this.active.values()) {
      if (this.pool.length < this.poolSize) {
        this.pool.push(item);
      }
    }
    this.active.clear();
  }

  update(keys: Tensor, values: Tensor, layerIdx: number): void {
    if (!this.active.has(layerIdx) && this.pool.length > 0) {
      this.pool.pop();
    }
    this.active.set(layerIdx, { keys, values });
  }

  get(layerIdx: number): { keys: Tensor; values: Tensor } | null {
    return this.active.get(layerIdx) || null;
  }
}

export class State {
  graph: Graph;
  kvCache: KVCache;
  currentLength: number;
  isPrefill: boolean;

  constructor(graph: Graph, kvCache: KVCache) {
    this.graph = graph;
    this.kvCache = kvCache;
    this.currentLength = 0;
    this.isPrefill = true;
  }

  /** Reset state for a new generation */
  reset(): void {
    this.kvCache.clear();
    this.currentLength = 0;
    this.isPrefill = true;
  }
}

export class MultiHeadAttentionCache implements KVCache {
  private cache: Map<number, { keys: Tensor; values: Tensor }> = new Map();
  private numHeads: number;
  private headDim: number;

  constructor(numHeads: number, headDim: number) {
    this.numHeads = numHeads;
    this.headDim = headDim;
  }

  clear(): void {
    this.cache.clear();
  }

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

  get(layerIdx: number): { keys: Tensor; values: Tensor } | null {
    return this.cache.get(layerIdx) || null;
  }
}

export class GroupedQueryAttentionCache implements KVCache {
  private cache: Map<number, { keys: Tensor; values: Tensor }> = new Map();
  private numKVHeads: number;
  private headDim: number;

  constructor(numKVHeads: number, headDim: number) {
    this.numKVHeads = numKVHeads;
    this.headDim = headDim;
  }

  clear(): void {
    this.cache.clear();
  }

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

  get(layerIdx: number): { keys: Tensor; values: Tensor } | null {
    return this.cache.get(layerIdx) || null;
  }
}

export class MultiQueryAttentionCache implements KVCache {
  private cache: Map<number, { keys: Tensor; values: Tensor }> = new Map();
  private numKVHeads: number = 1;
  private headDim: number;

  constructor(headDim: number) {
    this.headDim = headDim;
  }

  clear(): void {
    this.cache.clear();
  }

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

  get(layerIdx: number): { keys: Tensor; values: Tensor } | null {
    return this.cache.get(layerIdx) || null;
  }
}

export class SequenceBatchingKVCache implements KVCache {
  private cache: Map<number, { keys: Tensor; values: Tensor }[]> = new Map();

  clear(): void {
    this.cache.clear();
  }

  update(keys: Tensor, values: Tensor, layerIdx: number): void {
    if (!this.cache.has(layerIdx)) {
      this.cache.set(layerIdx, []);
    }
    this.cache.get(layerIdx)!.push({ keys, values });
  }

  get(layerIdx: number): { keys: Tensor; values: Tensor } | null {
    const batches = this.cache.get(layerIdx);
    if (!batches || batches.length === 0) {
      return null;
    }
    return batches[batches.length - 1]!;
  }
}

export class CrossAttentionCache implements KVCache {
  private cache: Map<number, { keys: Tensor; values: Tensor }> = new Map();

  clear(): void {
    this.cache.clear();
  }

  update(keys: Tensor, values: Tensor, layerIdx: number): void {
    this.cache.set(layerIdx, { keys, values });
  }

  get(layerIdx: number): { keys: Tensor; values: Tensor } | null {
    return this.cache.get(layerIdx) || null;
  }
}

export class SlidingWindowKVCache implements KVCache {
  private cache: Map<number, { keys: Tensor; values: Tensor }> = new Map();
  private windowSize: number;

  constructor(windowSize: number) {
    this.windowSize = windowSize;
  }

  clear(): void {
    this.cache.clear();
  }

  update(keys: Tensor, values: Tensor, layerIdx: number): void {
    const seqLen = keys.shape.length > 2 ? (keys.shape[2] as number) : (keys.shape[1] as number);
    if (seqLen > this.windowSize) {
      // Implement sliding window truncation here via view logic
    }
    this.cache.set(layerIdx, { keys, values });
  }

  get(layerIdx: number): { keys: Tensor; values: Tensor } | null {
    return this.cache.get(layerIdx) || null;
  }
}

export class PositionalEmbeddingUtils {
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

  static applyALiBi(attentionScores: Tensor, numHeads: number): Tensor {
    if (!(attentionScores.data instanceof Float32Array)) {
      return attentionScores;
    }

    const data = attentionScores.data as Float32Array;
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

export class QuantizedKVCache implements KVCache {
  constructor(private dtype: string = 'int8') {}
  clear(): void {}
  update(keys: Tensor, values: Tensor, layerIdx: number): void {}
  get(layerIdx: number): { keys: Tensor; values: Tensor } | null {
    return null;
  }
}

export class OffloadedKVCache implements KVCache {
  constructor(private maxVramSize: number) {}
  clear(): void {}
  update(keys: Tensor, values: Tensor, layerIdx: number): void {}
  get(layerIdx: number): { keys: Tensor; values: Tensor } | null {
    return null;
  }
}

export class PromptCacheManager {
  async saveToIDB(prompt: string, state: any): Promise<void> {}
}
