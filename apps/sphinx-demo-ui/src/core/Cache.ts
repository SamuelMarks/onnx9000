/* eslint-disable */
// @ts-nocheck
/**
 * A simple TTL-based eviction cache for ASTs and WASM outputs.
 */
export class Cache<T> {
  private store: Map<string, { value: T; expiresAt: number }> = new Map();
  private defaultTtlMs: number;

  constructor(defaultTtlMs: number = 60000) {
    this.defaultTtlMs = defaultTtlMs;
  }

  public set(key: string, value: T, customTtlMs?: number): void {
    const ttl = customTtlMs ?? this.defaultTtlMs;
    this.store.set(key, { value, expiresAt: Date.now() + ttl });
  }

  public get(key: string): T | undefined {
    const item = this.store.get(key);
    if (!item) return undefined;

    if (Date.now() > item.expiresAt) {
      this.store.delete(key);
      return undefined;
    }

    return item.value;
  }

  public clear(): void {
    this.store.clear();
  }

  public evictExpired(): void {
    const now = Date.now();
    for (const [key, item] of this.store.entries()) {
      if (now > item.expiresAt) {
        this.store.delete(key);
      }
    }
  }
}
