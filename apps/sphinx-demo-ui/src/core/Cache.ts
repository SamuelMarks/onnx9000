/**
 * @fileoverview Cache module for sphinx-demo-ui.
 */
/**
 * A simple TTL-based eviction cache for ASTs and WASM outputs.
 */
export class Cache<T> {
  /** Property `store` of `Cache`. */
  private store: Map<string, { value: T; expiresAt: number }> = new Map();
  /** Property `defaultTtlMs` of `Cache`. */
  private defaultTtlMs: number;

  /**
   * Initializes a new `Cache` instance.
   * @param defaultTtlMs Parameter `defaultTtlMs`.
   */
  constructor(defaultTtlMs: number = 60000) {
    this.defaultTtlMs = defaultTtlMs;
  }

  /**
   * Executes `set` on `Cache`.
   * @param key Parameter `key`.
   * @param value Parameter `value`.
   * @param customTtlMs Parameter `customTtlMs`.
   */
  public set(key: string, value: T, customTtlMs?: number): void {
    const ttl = customTtlMs ?? this.defaultTtlMs;
    this.store.set(key, { value, expiresAt: Date.now() + ttl });
  }

  /**
   * Executes `get` on `Cache`.
   * @param key Parameter `key`.
   * @returns Result of `get`.
   */
  public get(key: string): T | undefined {
    const item = this.store.get(key);
    if (!item) return undefined;

    if (Date.now() > item.expiresAt) {
      this.store.delete(key);
      return undefined;
    }

    return item.value;
  }

  /** Executes `clear` on `Cache`. */
  public clear(): void {
    this.store.clear();
  }

  /** Executes `evictExpired` on `Cache`. */
  public evictExpired(): void {
    const now = Date.now();
    for (const [key, item] of this.store.entries()) {
      if (now > item.expiresAt) {
        this.store.delete(key);
      }
    }
  }
}
