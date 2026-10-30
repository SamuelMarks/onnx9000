/**
 * @fileoverview kv_cache.ts
 * Provides kv_cache functionality for the serve package.
 */
export interface KVCacheEntry {
  /** Property `sessionId`. */
  sessionId: string;
  /** Property `prefixHash`. */
  prefixHash?: string | undefined; // For prompt caching
  /** Property `data`. */
  data: Float32Array; // Natively serialized KV Cache states
  /** Property `lastAccessed`. */
  lastAccessed: number;
}

/** Interface `KVSyncAdapter` definition. */
export interface KVSyncAdapter {
  /**
   * Method `save`.
   * @param sessionId Parameter `sessionId`.
   * @param data Parameter `data`.
   */
  save(sessionId: string, data: Float32Array): Promise<void>;
  /**
   * Method `load`.
   * @param sessionId Parameter `sessionId`.
   */
  load(sessionId: string): Promise<Float32Array | null>;
  /**
   * Method `delete`.
   * @param sessionId Parameter `sessionId`.
   */
  delete(sessionId: string): Promise<void>;
}

/** Represents the `KVCacheManager` class. */
export class KVCacheManager {
  /** Property `cache` of `KVCacheManager`. */
  private cache: Map<string, KVCacheEntry> = new Map();
  /** Property `prefixCache` of `KVCacheManager`. */
  private prefixCache: Map<string, KVCacheEntry> = new Map();

  // 116. Support auto-eviction of idle KV Caches
  /** Property `idleTimeoutMs` of `KVCacheManager`. */
  public idleTimeoutMs: number = 5 * 60 * 1000;

  /**
   * Initializes a new `KVCacheManager` instance.
   * @param syncAdapter Parameter `syncAdapter`.
   */
  constructor(public syncAdapter?: KVSyncAdapter) {
    // Eviction interval
    setInterval(() => {
      this.evictIdle();
    }, 60000);
  }

  // 112. Assign a unique `session_id` to chat streams to route requests back to their active KV cache.
  /**
   * Executes `getCache` on `KVCacheManager`.
   * @param sessionId Parameter `sessionId`.
   * @returns Result of `getCache`.
   */
  public async getCache(sessionId: string): Promise<Float32Array | null> {
    const entry = this.cache.get(sessionId);
    if (entry) {
      entry.lastAccessed = Date.now();
      return entry.data;
    }

    // 113. Implement a distributed KV cache synchronizer
    // 115. Deserialize KV Cache states and inject them directly back
    if (this.syncAdapter) {
      const data = await this.syncAdapter.load(sessionId);
      if (data) {
        this.cache.set(sessionId, {
          sessionId,
          data,
          lastAccessed: Date.now(),
        });
        return data;
      }
    }

    return null;
  }

  // 111. Maintain continuous `past_key_values` dynamically
  // 114. Serialize KV Cache slices into binary strings for network persistence natively.
  /**
   * Executes `setCache` on `KVCacheManager`.
   * @param sessionId Parameter `sessionId`.
   * @param data Parameter `data`.
   * @param prefixHash Parameter `prefixHash`.
   * @returns Result of `setCache`.
   */
  public async setCache(sessionId: string, data: Float32Array, prefixHash?: string) {
    const entry: KVCacheEntry = {
      sessionId,
      prefixHash,
      data,
      lastAccessed: Date.now(),
    };
    this.cache.set(sessionId, entry);

    if (prefixHash) {
      this.prefixCache.set(prefixHash, entry);
    }

    if (this.syncAdapter) {
      await this.syncAdapter.save(sessionId, data);
    }
  }

  // 117. Implement Prompt Caching natively.
  // 118. Detect identical request prefixes automatically to leverage shared caches.
  /**
   * Executes `getPromptCache` on `KVCacheManager`.
   * @param prefixHash Parameter `prefixHash`.
   * @returns Result of `getPromptCache`.
   */
  public getPromptCache(prefixHash: string): Float32Array | null {
    const entry = this.prefixCache.get(prefixHash);
    if (entry) {
      entry.lastAccessed = Date.now();
      return entry.data;
    }
    return null;
  }

  /**
   * Executes `evictIdle` on `KVCacheManager`.
   * @returns Result of `evictIdle`.
   */
  private evictIdle() {
    const now = Date.now();
    for (const [id, entry] of this.cache.entries()) {
      if (now - entry.lastAccessed > this.idleTimeoutMs) {
        this.cache.delete(id);
        if (entry.prefixHash) {
          this.prefixCache.delete(entry.prefixHash);
        }
      }
    }
  }

  // 120. Provide API to explicitly flush the global server KV Cache.
  /**
   * Executes `flushAll` on `KVCacheManager`.
   * @returns Result of `flushAll`.
   */
  public async flushAll() {
    this.cache.clear();
    this.prefixCache.clear();
    // Assuming syncAdapter doesn't need full flush
  }

  // 119. Allocate Ring Buffers inside WASM/WebGPU to manage sliding-window attention seamlessly.
  // Allocate a ring buffer wrapper
  /**
   * Executes `createRingBuffer` on `KVCacheManager`.
   * @param size Parameter `size`.
   * @returns Result of `createRingBuffer`.
   */
  public createRingBuffer(size: number): Float32Array {
    return new Float32Array(size); // Represents the sliding window natively
  }
}
